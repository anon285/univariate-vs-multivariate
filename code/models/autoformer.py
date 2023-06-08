import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pytorch_lightning_utilities import *
import evaluate_model
import math
import numpy as np

from models.other.Autoformer.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from models.other.Autoformer.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.other.Autoformer.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(PLInit, PLTrainer, PLAnalysis, PLMisc, evaluate_model.model_testing, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = self.hparams.seq_len
        self.label_len = self.hparams.label_len
        self.pred_len = self.hparams.pred_len
        self.output_attention = self.hparams.output_attention

        # Decomp
        kernel_size = self.hparams.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.hparams.enc_in, self.hparams.d_model,
                                                  self.hparams.embed, self.hparams.freq,
                                                  self.hparams.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.hparams.dec_in, self.hparams.d_model,
                                                  self.hparams.embed, self.hparams.freq,
                                                  self.hparams.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.hparams.factor,
                                        attention_dropout=self.hparams.dropout,
                                        output_attention=self.hparams.output_attention),
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
                        AutoCorrelation(True, self.hparams.factor,
                                        attention_dropout=self.hparams.dropout,
                                        output_attention=False),
                        self.hparams.d_model, self.hparams.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.hparams.factor,
                                        attention_dropout=self.hparams.dropout,
                                        output_attention=False),
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
        #print(f'x_enc.shape:\n  {x_enc.shape}')
        #print(f'x_mark_enc.shape:\n  {x_mark_enc.shape}')
        #print(f'x_dec.shape:\n  {x_dec.shape}')
        #print(f'x_mark_dec.shape:\n  {x_mark_dec.shape}')
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
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
            outputs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

        f_dim = -1 if self.hparams.features == 'MS' else 0
        outputs = outputs[:, -self.hparams.pred_len:, f_dim:]
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
        parser = parent_parser.add_argument_group("Autoformer model")
        # basic config
        parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
        parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')
        # data loader
        parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/',
                            help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
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
        parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
        parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25,
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
