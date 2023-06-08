import pytorch_lightning as pl
from models.pytorch_lightning_utilities import *
import evaluate_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.other.FEDformer_.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from models.other.FEDformer_.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.other.FEDformer_.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from models.other.FEDformer_.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from models.other.FEDformer_.layers.SelfAttention_Family import FullAttention, ProbAttention
from models.other.FEDformer_.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FEDformer(PLInit, PLTrainer, PLAnalysis, PLMisc, evaluate_model.model_testing, pl.LightningModule):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, *args, **kwargs):
        super(FEDformer, self).__init__(*args, **kwargs)
        self.version = self.hparams.version
        self.mode_select = self.hparams.mode_select
        self.modes = self.hparams.modes
        self.seq_len = self.hparams.seq_len
        self.label_len = self.hparams.label_len
        self.pred_len = self.hparams.pred_len
        self.output_attention = self.hparams.output_attention

        # Decomp
        kernel_size = self.hparams.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.hparams.enc_in, self.hparams.d_model, self.hparams.embed, self.hparams.freq,
                                                  self.hparams.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.hparams.dec_in, self.hparams.d_model, self.hparams.embed, self.hparams.freq,
                                                  self.hparams.dropout)

        if self.hparams.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=self.hparams.d_model, L=self.hparams.L, base=self.hparams.base)
            decoder_self_att = MultiWaveletTransform(ich=self.hparams.d_model, L=self.hparams.L, base=self.hparams.base)
            decoder_cross_att = MultiWaveletCross(in_channels=self.hparams.d_model,
                                                  out_channels=self.hparams.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.hparams.modes,
                                                  ich=self.hparams.d_model,
                                                  base=self.hparams.base,
                                                  activation=self.hparams.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=self.hparams.d_model,
                                            out_channels=self.hparams.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.hparams.modes,
                                            mode_select_method=self.hparams.mode_select)
            decoder_self_att = FourierBlock(in_channels=self.hparams.d_model,
                                            out_channels=self.hparams.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=self.hparams.modes,
                                            mode_select_method=self.hparams.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=self.hparams.d_model,
                                                      out_channels=self.hparams.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.hparams.modes,
                                                      mode_select_method=self.hparams.mode_select)
        # Encoder
        enc_modes = int(min(self.hparams.modes, self.hparams.seq_len//2))
        dec_modes = int(min(self.hparams.modes, (self.hparams.seq_len//2+self.hparams.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.hparams.d_model, self.hparams.n_heads),

                    self.hparams.d_model,
                    self.hparams.d_ff,
                    moving_avg=self.hparams.moving_avg,
                    dropout=self.hparams.dropout,
                    activation=self.hparams.activation
                ) for l in range(self.hparams.e_layers)
            ],
            norm_layer=my_Layernorm(self.hparams.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        self.hparams.d_model, self.hparams.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        self.hparams.d_model, self.hparams.n_heads),
                    self.hparams.d_model,
                    self.hparams.c_out,
                    self.hparams.d_ff,
                    moving_avg=self.hparams.moving_avg,
                    dropout=self.hparams.dropout,
                    activation=self.hparams.activation,
                )
                for l in range(self.hparams.d_layers)
            ],
            norm_layer=my_Layernorm(self.hparams.d_model),
            projection=nn.Linear(self.hparams.d_model, self.hparams.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


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

  #--is_training 1 \
  #--root_path ./dataset/ETT-small/ \
  #--data_path ETTm1.csv \
  #--task_id ETTm1 \
  #--model $model \
  #--data ETTm1 \
  #--features M \
  #--seq_len 96 \
  #--label_len 48 \
  #--pred_len $preLen \
  #--e_layers 2 \
  #--d_layers 1 \
  #--factor 3 \
  #--enc_in 7 \
  #--dec_in 7 \
  #--c_out 7 \
  #--des 'Exp' \
  #--d_model 512 \
  #--itr 3 \

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Autoformer model")
        # basic config
        #parser.add_argument('--task_id', type=str, required=False, default='test', help='model id')
        parser.add_argument('--model', type=str, required=False, default='FEDformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')
        # data loader
        parser.add_argument('--data', type=str, default='custom', help='dataset type')
        parser.add_argument('--root_path', type=str, default='../data/autoformer/electricity/',
                            help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict '+\
                            'multivariate, S:univariate predict univariate, '+\
                            'MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, '+\
                            't:minutely, h:hourly, d:daily, b:business days, w:weekly, '+\
                            'm:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                            help='location of model checkpoints')
        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        # model define
        #parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
        #parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=24,
                            help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means'+
                            ' not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true',
                            help='whether to output attention in encoder')
        parser.add_argument('--do_predict', action='store_true',
                            help='whether to predict unseen future data')
        #parser.add_argument('--des', type=str, default='Exp', help='exp description')
        parser.add_argument('--version', type=str, default='Fourier',
                            help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
        parser.add_argument('--mode_select', type=str, default='random',
                            help='for FEDformer, there are two mode selection method, options: [random, low]')
        parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')


if __name__ == '__main__':
    class config(object):
        ab = 0
        modes = 32
        mode_select = 'random'
        # version = 'Fourier'
        version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0

    config = config()
    model = Model(config)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([3, config.seq_len, 7])
    enc_mark = torch.randn([3, config.seq_len, 4])

    dec = torch.randn([3, config.seq_len//2+config.pred_len, 7])
    dec_mark = torch.randn([3, config.seq_len//2+config.pred_len, 4])
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print(out)
