"""Pytorch Lightning implementation of full attention 
"""

import pytorch_lightning as pl
from utilities import decorators
import torch
from models import pl_init, PLMisc, PLTrainer

class FullAttention(pl.LightningModule):
    """Full attention.
    """
    def __init__(self, hparams):
        super().__init__()
        # Project the embeding down to Q, K, V
        self.save_hyperparameters(hparams)

        self.query_projection = torch.nn.Linear(
            hparams.model_dim, hparams.head_dim*self.hparams.num_heads)
        self.key_projection = torch.nn.Linear(
            hparams.model_dim, hparams.head_dim*self.hparams.num_heads)
        self.value_projection = torch.nn.Linear(
            hparams.model_dim, hparams.head_dim*self.hparams.num_heads)
        if hparams.use_advanced_embedding:
            self.embedding_layer = AdvancedEmbedding(hparams)
        else:
            self.embedding_layer = BasicEmbedding(hparams)

        self.dropout = torch.nn.Dropout(hparams.dropout)

    def attention_scores(self):
        pass

    def separate_heads(self, merged):
        """To simplify the code the embedding has been projected into one vector with all the heads
        effectively concatenated.  We now need to separate the heads.
        """
        # merged.shape: (b, l, merged_dim==model_dim)
        # unmerged.shape: (b, l, h, head_dim)
        print('Unmerging heads...')
        print(merged.shape)
        unmerged = merged.view(merged.shape[0:2] + (self.hparams.num_heads, self.hparams.head_dim))
        # Swap the axis so that the attention scores can be calculated with matrix multiplication
        # (b, l, num_heads, head_dim) --> (b, num_heads, l, head_dim)
        unmerged = unmerged.permute(0, 2, 1, 3)
        print(unmerged.shape)
        return unmerged

    def merge_heads(self, unmerged):
        """All the heads from the layer need to be merged to form the context to be fed into a
        decoder.  In the illustrated transformer the heads are concatenated and then projected to
        the model dim.  In our case, after concatination, the dimension == model dim
        """
        # unmerged.shape: (b, l, h, head_dim)
        # merged.shape: (b, l, merged_dim==model_dim)
        print('Merging heads...')
        print(unmerged.shape)
        # Reset the axis order
        # (b, num_heads, l, head_dim) --> (b, l, num_heads, head_dim)
        merged = unmerged.permute(0, 2, 1, 3).contiguous()
        merged = merged.view(merged.shape[0:2] + (merged.shape[2]*merged.shape[3],))
        print(merged.shape)
        return merged

    def forward(self, embed):
        """embed.shape: (B, L, D)
        out.shape: (B, L, D)
        """
        # The heads are all concatenated, we need to separate them
        query = self.separate_heads(self.query_projection(embed))
        key   = self.separate_heads(self.key_projection(embed))
        value = self.separate_heads(self.value_projection(embed))

        print('Attention matmul...')
        print(query.shape)
        attention_score = torch.matmul(query, key.transpose(-1, -2))
        print(attention_score.shape)
        print('    should == (100, 4, 500, 500)')
        attention_score /= self.hparams.head_dim**0.5

        attention_prob = torch.nn.functional.softmax(attention_score)
        attention_prob = self.dropout(attention_prob)

        print('value vector scailing, shape should be maintained...')
        print(value.shape)
        context = torch.matmul(attention_prob, value)
        print(value.shape)

        # We now have a context vector for each head.
        # The heads are separated so lets merge the heads
        context = self.merge_heads(context)
        if context.shape[2] != self.hparams.model_dim:
            # The dimension should now be back to model_dim
            raise ValueError(
                'Dimension of context tensor != model_dim\ncontext.shape:{}\nmodel_dim:{}'.format(
                     context.shape, self.hparams.model_dim))
        print(context.shape)
        breakpoint()
        return context


class FullAttentionModel(pl.LightningModule, PLTrainer, PLMisc):
    """ Simple, full attention, based on the implementation from hugging face big_bird
    """
    def __init__(self, hparams,  learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters(hparams, ignore='log')

        if hparams.model_dim % hparams.num_heads != 0:
            raise ValueError('num_heads: {self.hparams.num_heads} needs to be wholly divisible' +\
                             'by model_dim: {self.hparms.num_heads}')
        else:
            self.hparams.head_dim = hparams.model_dim // hparams.num_heads
        self.full_attention = FullAttention(self.hparams)

    def on_train_start(self, *args, **kwargs):
        return self.inherited_on_train_start(*args, **kwargs)

    def configure_optimizers(self, *args, **kwargs):
        return self.inherited_configure_optimizers(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        return self.inherited_training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.inherited_validation_ste(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.inherited_test_step(*args, **kwargs)


    def forward(self, x):
        print(x.shape)
        context = self.full_attention(x)
        return context

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Full attention")
        parser.add_argument(
            '--model_dim', type=int, default=64,
            help='model_dim: model dimension')
        parser.add_argument(
            '--num_heads', type=int, default=8,
            help='number of heads within a transformer layer')
        parser.add_argument(
            '--dropout', type=float, default=0.1,
            help='dropout probability')
        parser.add_argument(
            '--input_dim', type=int, default=1,
            help='dimension of single time step vector')

        return parent_parser
