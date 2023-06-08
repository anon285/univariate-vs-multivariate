import pytorch_lightning as pl
import torch
from models.pytorch_lightning_utilities import *
import evaluate_model


import torch.nn as nn
from models.other.Informer2020_.models.embed import DataEmbedding
from models.other.Informer2020_.models.attn import FullAttention, ProbAttention, AttentionLayer
from models.other.Informer2020_.utils.masking import TriangularCausalMask, ProbMask
from models.other.Informer2020_.models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.other.Informer2020_.models.decoder import Decoder, DecoderLayer


class Informer(PLInit, PLTrainer, PLAnalysis, PLMisc, evaluate_model.model_testing, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pred_len = self.hparams.pred_len
        self.attn = self.hparams.attn
        self.output_attention = self.hparams.output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(self.hparams.enc_in, self.hparams.d_model, self.hparams.embed, self.hparams.freq, self.hparams.dropout)
        self.dec_embedding = DataEmbedding(self.hparams.dec_in, self.hparams.d_model, self.hparams.embed, self.hparams.freq, self.hparams.dropout)
        # Attention
        Attn = ProbAttention if self.attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, self.hparams.factor, attention_dropout=self.hparams.dropout, output_attention=self.hparams.output_attention), 
                                self.hparams.d_model, self.hparams.n_heads, mix=False),
                    self.hparams.d_model,
                    self.hparams.d_ff,
                    dropout=self.hparams.dropout,
                    activation=self.hparams.activation
                ) for l in range(self.hparams.e_layers)
            ],
            [
                ConvLayer(
                    self.hparams.d_model
                ) for l in range(self.hparams.e_layers-1)
            ] if self.hparams.distil else None,
            norm_layer=torch.nn.LayerNorm(self.hparams.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, self.hparams.factor, attention_dropout=self.hparams.dropout, output_attention=False), 
                                self.hparams.d_model, self.hparams.n_heads, mix=self.hparams.mix),
                    AttentionLayer(FullAttention(False, self.hparams.factor, attention_dropout=self.hparams.dropout, output_attention=False), 
                                self.hparams.d_model, self.hparams.n_heads, mix=False),
                    self.hparams.d_model,
                    self.hparams.d_ff,
                    dropout=self.hparams.dropout,
                    activation=self.hparams.activation,
                )
                for l in range(self.hparams.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.hparams.d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+pred_len, out_channels=pred_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(self.hparams.d_model, self.hparams.c_out, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


    def training_step(self, batch, batch_idx, evaluate=False):
        if self.hparams.log:
            if self.trainer.global_step != 0:
                if self.trainer.global_step % self.hparams.validate_every == 0:
                    self.train_loss(batches=self.hparams.validate_batches)
                    self.val_loss(batches=self.hparams.validate_batches)
                    self.test_loss(batches=self.hparams.validate_batches)

        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(self.device)

        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.hparams.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.hparams.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.hparams.output_attention:
            outputs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.hparams.features == 'MS' else 0
        batch_y = batch_y[:, -self.hparams.pred_len:, f_dim:].to(self.device)
        loss = F.mse_loss(outputs, batch_y)

        if evaluate:
            mae_loss = F.l1_loss(outputs, batch_y)
            loss_per_dim_mse = self.loss_per_dim_mse_func(outputs, batch_y)
            loss_per_dim_mse = loss_per_dim_mse.mean(dim=[0,1]).detach().cpu().numpy()
            loss_per_dim_mae = self.loss_per_dim_mae_func(outputs, batch_y)
            loss_per_dim_mae = loss_per_dim_mae.mean(dim=[0,1]).detach().cpu().numpy()
            return loss, mae_loss, loss_per_dim_mse, loss_per_dim_mae
        else:
            return loss




    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Informer")

        parser.add_argument('--data', type=str, required=True, default='custom', help='data')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
        parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        #parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
        parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
        parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
        # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu',help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        #parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
        parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
        parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
        #parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        #parser.add_argument('--itr', type=int, default=2, help='experiments times')
        #parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
        #parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        #parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        #parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test',help='exp description')
        parser.add_argument('--loss', type=str, default='mse',help='loss function')
        #parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

        #parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        #parser.add_argument('--gpu', type=int, default=0, help='gpu')
        #parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        #parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
