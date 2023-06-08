"""Embedding for time series.  Also includes the decoder
"""

import torch
import pytorch_lightning as pl


class BasicEmbedding(pl.LightningModule):
    """This will embed the input time series using basic CNN kernals.
    Args:
        self.hparams.model_dim
        self.hparams.kernal_size
    """
    def __init__(self, hparams):
        super().__init__()
        self.cnn = torch.nn.Conv1d(
            hparams.input_dim,
            hparams.model_dim,
            hparams.kernal_size,
            padding='same')

    def forward(self, x):
        # x.shape: (b, l, input_dim)
        # cnn input: (b, input_dim, l)
        x = x.permute(0, 2, 1)
        y = self.cnn(x)
        y = y.permute(0, 2, 1)
        return y


class LearntPositionalEncoding(pl.LightningModule):
    """Each position is a learnable vector that gets added to the input
    The is an independant time encoding for every time step in the history.
    This class does not make any simplifications
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.position_embeddings = torch.nn.Embedding(hparams.history_length, hparams.model_dim)
        self.idx = torch.arange(hparams.history_length).to(self.hparams.device)
        self.LayerNorm = torch.nn.LayerNorm(hparams.model_dim, eps=hparams.layer_norm_eps)
        self.dropout = torch.nn.Dropout(hparams.dropout)

    def something(self):
        """We need to change x.shape: (b, length, model_dim) to:
        (b, days, window_length, model_dim)
        """

    def forward(self, x):
        position_embeddings = self.position_embeddings(self.idx)
        x += position_embeddings
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x


class BasicDecoder(pl.LightningModule):
    """This will just be a simple feed forward layer for now
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.dense = torch.nn.Linear(hparams.model_dim*hparams.history_length,
                               hparams.horizon_length*hparams.input_dim)

    def forward(self, x):
        # x.shape: (B, L, d)  -->  (B, L*d)
        # Reshape the tensor so that all the timestep context vectors are in one dimension
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        out = self.dense(x)
        # Reshape to (B, L, d) where d is the dimension of the input data and not the model dim
        out = out.view(x.shape[0], self.hparams.horizon_length, self.hparams.input_dim)
        return out
