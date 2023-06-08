import pytorch_lightning as pl
import torch
from models import embedding
from models.pytorch_lightning_utilities import *
import evaluate_model


class VanillaTransformer(PLInit, PLTrainer, PLAnalysis, PLMisc, evaluate_model.model_testing, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.hparams.model_dim,
                                                         nhead=self.hparams.num_heads,
                                                         batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer,
                                                 num_layers=self.hparams.num_layers)
        self.embedding = getattr(embedding, self.hparams.embedding)(self.hparams)
        self.positional_encoding = embedding.LearntPositionalEncoding(self.hparams)
        self.decoder = embedding.BasicDecoder(self.hparams)

    def add_hyperparameters(self):
        """This is automatically called from __init__
        """
        # Find input dim
        batch = next(iter(self.dataloaders['train']))
        self.hparams.input_dim = batch[0].shape[-1]

    def forward(self, x):
        x.requires_grad = True
        embed = self.embedding(x)
        embed = self.positional_encoding(embed)
        context = self.transformer_encoder(embed)
        x_hat = self.decoder(context)
        return x_hat

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Seasonal Transformer model")
        parser.add_argument(
            '--model_dim', type=int, default=64,
            help='model_dim: model dimension')
        parser.add_argument(
            '--num_layers', type=int, default=6,
            help='number of layers of encoders to use')
        parser.add_argument(
            '--num_heads', type=int, default=4,
            help='number of heads within a transformer layer')
        parser.add_argument(
            '--kernal_size', type=int, default=7,
            help='size of the kernals in basic CNN embedding')
        parser.add_argument(
            '--layer_norm_eps', type=float, default=1e-12,  # The pytorch default is 1e-5
            help='Eps used for normalisation.  The PyTorch default is 1e-5 but Hugging Face'+\
                'use 1e-12 for some reason.')
        parser.add_argument(
            '--dropout', type=float, default=0.1,
            help='dropout probability')
        parser.add_argument(
            '--linear_embedding', action='store_true', default=False,
            help='Linear embedding instead of CNN embedding')
        parser.add_argument(
            '--linear_windowed_embedding', action='store_true', default=False,
            help='Linear windowed embedding instead of CNN embedding, set to window size or 0' +
            ' to disable')
        parser.add_argument(
            '--dimensional_cnn_embedding', action='store_true', default=False,
            help='Linear windowed embedding instead of CNN embedding, set to window size or 0' +
            ' to disable')
