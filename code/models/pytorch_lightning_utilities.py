"""Base classes defining common methods used in pytorch lightning models
"""
# pylint: disable=[E0602, E1101, E1102]
import pytorch_lightning as pl
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from tqdm import tqdm


# DELME - no longer used
def pl_init(cls):
    """A decorator that adds the boiler plate stuff needed in the __init__
    I think the learning_rate stuff is needed for learning rate finder
    """
    def new_init(self, hparams,  learning_rate=1e-3, **kwargs):
        super(cls, cls).__init__(self)  # This seem janky as fuck but it works
        self.learning_rate = learning_rate
        self.save_hyperparameters(hparams, ignore='log')
        breakpoint()
        cls.init(*args, kwargs)

    cls.__init__ = new_init
    return cls

class PLInit():
    def __init__(self, hparams, dataloaders, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.dataloaders = dataloaders
        self.best_loss = {'train': {'mse': 999, 'mae': 999, 'mse_per_dim': 999,
                                    'mae_per_dim': 999, 'model': []},
                          'val': {'mse': 999, 'mae': 999, 'mse_per_dim': 999,
                                    'mae_per_dim': 999, 'model': []},
                          'test': {'mse': 999, 'mae': 999, 'mse_per_dim': 999,
                                    'mae_per_dim': 999, 'model': []},
                          'experiment': {'mse': 999, 'mae': 999, 'mse_per_dim': 999,
                                    'mae_per_dim': 999}}
        super().__init__()
        self.loss_per_dim_mse_func = torch.nn.MSELoss(reduction='none')
        self.loss_per_dim_mae_func = torch.nn.L1Loss(reduction='none')
        self.save_hyperparameters(hparams, ignore='log')
        self.add_hyperparameters()

    def add_hyperparameters(self):
        """This can be overloaded in the model child class if needed
        """
        # Find input dim
        batch = next(iter(self.dataloaders['train']))
        self.hparams.input_dim = batch[0].shape[-1]

class PLMisc():
    def on_train_start(self):
        if self.hparams.log:
            self.logger.log_hyperparams(self.hparams, {'best/train_loss': 0,
                                                       'best/val_loss': 0,
                                                       'best/test_loss': 0,
                                                       'best/experiment_loss': 0,
                                                       'train_loss': 0,
                                                       'val_loss': 0,
                                                       'test_loss': 0})

    def configure_optimizers(self):
        """StepLR() args:
                step_size(int) - Period of learning rate decay
                gamma(float) - Multiplicative factor of learning rate decay, default: 0.1
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate))
        if 'optimizer_step_size' in self.hparams:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.optimizer_step_size,
                gamma=self.hparams.optimizer_gamma,
                verbose=True)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            return {'optimizer': optimizer}


class PLAnalysis():
    def visualise_losses_hist(self, x_hat, y, bins=20):
        losses = F.mse_loss(x_hat, y, reduction='none')
        losses = losses.mean(dim=1)

        s = pd.Series(losses.cpu().detach())
        ax = s.plot.hist(bins=bins)

        ax.set(xlabel='Loss', ylabel='Frequency', title='Histogram of Losses of Minibatch')
        plt.show()

    def visualise_losses_density(self, x_hat, y):
        losses = F.mse_loss(x_hat, y, reduction='none')
        losses = losses.mean(dim=1)

        s = pd.Series(losses.cpu().detach())
        ax = s.plot.kde()

        ax.set(xlabel='Loss', ylabel='Density', title='Kernal Density Estimation of Minibatch')
        plt.show()

    def input_differentiation(self, mode, plot=False):
        """Differentiate the output with respect to each input and then sum each dimension.
        This will show the impact each input dimension has on the output forecast
        args:
            dim: the selected dimension of the forecast
        """
        self.load_state_dict(self.best_loss[mode]['model'])
        # After training the model seems to be moved back to CPU?
        self.to(self.hparams.device)

        it = iter(self.dataloaders['test'])
        batch = next(it)
        #batch = next(iter(self.dataloaders['test']))
        x = batch[0]  # shape: (20, 16, 7)
        x = x.to(self.hparams.device)
        x.requires_grad=True

        x_hat = self(x)  # x_hat.shape: (batch, length, dims)
        #x_hat = torch.mean(x_hat, dim=(0,1))  # x_hat.shape: (dims)
        diff = torch.zeros(size=(x_hat.shape[-1], x_hat.shape[-1]), device=self.hparams.device)
        num_loops = x_hat.shape[0] * x_hat.shape[1]  # batches * horizon
        for batch_num, forecast in enumerate(tqdm(x_hat)):
            for time_step in forecast:
                diff_dimension = []
                for dim in time_step:
                    dim.backward(retain_graph=True)
                    # Calculate grad of average absolute impact each input dim has on this output
                    # dimension of a time step
                    grad = torch.mean(torch.abs(x.grad[batch_num]), dim=0)  # Average the length
                    diff_dimension.append(grad)
                    x.grad.data.zero_()
                diff_dimension = torch.stack(diff_dimension)
                diff = diff + diff_dimension / num_loops
                # We are adding a time step to the list.  We could instead divide and add
        # diff.shape: (batch, horizon, dim, dim)
        # We need to average the batch and horizon
        diff = diff.cpu().numpy()
        # diff.shape: (dims, dims)

        # For example the first row of diff shows how each dimension of the input affects the output
        # forecast of the first dimension
        if plot:
            plt.matshow(diff)
            #plt.imshow(diff)
            plt.gca().xaxis.set_label_position('top')
            plt.xlabel('Input dims')
            plt.ylabel('Output dims')
            plt.savefig('delme4.svg',
                        format='svg',
                        bbox_inches='tight')
            plt.close()

        return diff


class PLTrainer():
    def training_step(self, batch, batch_idx, evaluate=False):
        """Args:
                batch:
                batch_idx:
                evaluate: If true MAE loss is also returned
        """
        # Run a validation test every n steps
        if self.hparams.log:
            if self.trainer.global_step != 0:
                if self.trainer.global_step % self.hparams.validate_every == 0:
                    self.train_loss(batches=self.hparams.validate_batches)
                    self.val_loss(batches=self.hparams.validate_batches)
                    self.test_loss(batches=self.hparams.validate_batches)

        x, y, stats = batch
        x = x.to(self.hparams.device)
        y = y.to(self.hparams.device)

        x_hat = self(x)

        loss = F.mse_loss(x_hat, y)
        if evaluate:
            mae_loss = F.l1_loss(x_hat, y)
            loss_per_dim_mse = self.loss_per_dim_mse_func(x_hat, y)
            loss_per_dim_mse = loss_per_dim_mse.mean(dim=[0,1]).detach().cpu().numpy()
            loss_per_dim_mae = self.loss_per_dim_mae_func(x_hat, y)
            loss_per_dim_mae = loss_per_dim_mae.mean(dim=[0,1]).detach().cpu().numpy()
            return loss, mae_loss, loss_per_dim_mse, loss_per_dim_mae
        else:
            return loss

    def validation_step(self, batch, batch_idx, evaluate=False):
        pass

    def test_step(self, batch, batch_idx, log=True, evaluate=False):
        pass


class MyEarlyStopping(pl.callbacks.early_stopping.EarlyStopping):
    def __init__(self, *args, **kwargs):
        self.global_step = 0
        super().__init__(*args, **kwargs)

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.global_step != 0:
            if self.global_step % pl_module.hparams.validate_every == 0:
                self._run_early_stopping_check(trainer)
        self.global_step += 1
