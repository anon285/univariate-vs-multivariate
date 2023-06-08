"""Evaluate the model
"""
import seaborn
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from abc import ABC
from torch.nn import functional as F
import os
import numpy as np
import copy

def unnormalise(x, stats, idx):
    return x * stats['std'][idx] + stats['mean'][idx]

def get_forecast(model, dataloader, args, num_forecasts):
    """Returns:
        x.shape: (num_forecasts, history, 1)
        x_hat.shape (num_forecasts, horizon)
        y.shape: (num_forecasts, horizon)
        stats: {'std':[], 'mean':[]} - used to unnormalise the time series
        loss.shape: (num_forecasts) - the loss of each forecast
    """
    torch.set_grad_enabled(False)
    model = model.to(args.device)
    model.eval()
    it = iter(dataloader)
    x_all =     torch.empty(size=(0,), device=args.device)
    x_hat_all = torch.empty(size=(0,), device=args.device)
    y_all =     torch.empty(size=(0,), device=args.device)
    stats_all = {'mean': torch.empty(size=(0,)),
                 'std': torch.empty(size=(0,)),
                 'name': [],
                 'offset': []}
    loss_all = []
    i = 0
    while num_forecasts > 0:
        if i >= len(it):
            print('End of epoch, no more forecasts')
            break
        i += 1
        x, y, stats = next(it)
        x = x.to(args.device)
        y = y.to(args.device)


        x_hat = model(x)

        x_all = torch.cat((x_all, x[0:num_forecasts]))
        x_hat_all = torch.cat((x_hat_all, x_hat[0:num_forecasts]))
        y_all = torch.cat((y_all, y[0:num_forecasts]))

        losses = F.mse_loss(x_hat, y, reduction='none')
        loss = losses.mean(dim=1)
        loss = loss[0:num_forecasts, 0]
        loss_all.extend(loss.cpu().detach().numpy())
        # Crop stats down to num_forecasts and add to stats_all
        for key in stats:
            if key == 'offset':
                stats[key] = [stats[key][0][0:num_forecasts], stats[key][1][0:num_forecasts]]
                if len(stats_all[key]) == 0:
                    stats_all[key] = stats[key]
                else:
                    stats_all[key] = [torch.cat((stats_all[key][0], stats[key][0])),
                                      torch.cat((stats_all[key][1], stats[key][1]))]
            else:
                stats[key] = stats[key][0:num_forecasts]
                if key == 'name':
                    stats_all[key].extend(stats[key])
                else:
                    stats_all[key] = torch.cat((stats_all[key], stats[key]))

        num_forecasts  -= x.shape[0]

    # Reduce batch to num_forecasts
    # x.shape:     [batch, history, 1] --> [history]
    # x_hat.shape: [batch, horizon]    --> [horizon]
    # y.shape:     [batch, horizon]    --> [hoirzon]
    x = x[0:num_forecasts,:,:].to('cpu').detach()
    x_hat = x_hat[0:num_forecasts,:].to('cpu').detach()
    y = y[0:num_forecasts,:].to('cpu').detach()

    # Move all tensors to the CPU
    x_all = x_all.cpu()
    x_hat_all = x_hat_all.cpu()
    y_all = y_all.cpu()
    stats_all['mean'] = stats_all['mean'].cpu()
    stats_all['std'] = stats_all['std'].cpu()
    stats_all['offset'][0] = stats_all['offset'][0].cpu()
    stats_all['offset'][1] = stats_all['offset'][1].cpu()

    model.train()
    torch.set_grad_enabled(True)

    return x_all, x_hat_all, y_all, stats_all, loss_all

def forecast(model, dataloader, args, show_fig=False, save_fig=True):
    print('Forecasting...')
    x, x_hat, y, stats, loss = get_forecast(model, dataloader, args, args.end)
    print('Saving plots...')
    for idx in range(args.end):
        # Unnormalise
        horizon = unnormalise(x_hat[idx], stats, idx)
        history = unnormalise(x[idx], stats, idx)
        ground_truth = unnormalise(y[idx], stats, idx)
        total_ground_truth = torch.cat([history, ground_truth])

        # Smooth joint between history and horizon
        horizon = torch.cat([history[-1:], horizon])
        #ground_truth = torch.cat([history[-1:], ground_truth])

        # Plot
        # Remove extra dimensions to the data
        total_ground_truth = total_ground_truth[:, 0]
        horizon = horizon[:, 0]
        history = history[:, 0]
        name = stats['name'][idx]
        offset = [stats['offset'][0][idx], stats['offset'][1][idx]]  # Not touched
        seaborn.set_theme(style='whitegrid')
        ax = seaborn.lineplot(
            label='Ground truth',
            x=range(len(total_ground_truth)),
            y=total_ground_truth)
        seaborn.lineplot(
            label='Forecast',
            x=range(len(history)-1, len(history)+len(horizon)-1),
            y=horizon)
        ax.set(
            xlabel='Time',
            ylabel='kW',
            title=f'Forecast - ({name}, {offset[0]}:{offset[1]}), (Loss: {loss[idx]:.3f})')
        plt.axvline(x=len(history)-1, color='black', linestyle='dashed')

        if save_fig is True:
            directory = '/home/will/Documents/period_transformer/code/figures/' +\
                args.experiment + '/' + str(model.logger.version) + '/'
            if os.path.isdir(directory) is False:
                os.makedirs(directory)
            copy_idx = 0
            file_name = 'plot_' + str(copy_idx)
            extension = '.svg'
            while os.path.isfile(directory+file_name+extension):
                copy_idx += 1
                file_name = 'plot_' + str(copy_idx)


            plt.savefig(directory+file_name+extension,
                        format='svg',
                        bbox_inches='tight',)
        if show_fig is True:
            plt.show()
        plt.close()


class model_testing(ABC):
    """The following set of functions are used test a model over many batches and return the
    average loss

    Args:
        model: The model to be tested.
        batches: Number of batches to test with.
        dataloader: Pytorch dataloader to be used.
        model_step: The function of the model to be called for inference.

        dataloaders: Dictionary of the train, val and test dataloaders
    """
    # pylint: disable=no-member
    def loss_base(self, batches, dataloader, model_step, loading_bar):
        # Temporarily stop logging
        logging_state = self.hparams.log
        self.hparams.log = False
        self.eval()
        with torch.no_grad():
            losses = {'mse': [], 'mae': [], 'mse_per_dim': [], 'mae_per_dim': []}
            for idx, batch in enumerate(tqdm(dataloader, total=batches, disable=not loading_bar)):
                mse, mae, mse_per_dim, mae_per_dim = model_step(batch, idx, evaluate=True)
                losses['mse'].append(mse)
                losses['mae'].append(mae)
                losses['mse_per_dim'].append(mse_per_dim)
                losses['mae_per_dim'].append(mae_per_dim)
                if idx == batches:
                    break
            mse = sum(losses['mse'])/len(losses['mse'])
            mae = sum(losses['mae'])/len(losses['mae'])
            mse_per_dim = sum(losses['mse_per_dim'])/len(losses['mse_per_dim'])
            mae_per_dim = sum(losses['mae_per_dim'])/len(losses['mae_per_dim'])
        self.train()
        self.hparams.log = logging_state

        return mse, mae, mse_per_dim, mae_per_dim

    def save_checkpoint(self, name, mode):
        """Name - the name of the file to be saved I think
        Mode - train, val or test
        """
        if self.hparams.checkpoint and name is not None:
            if mode == 'val':
                save_checkpoint_to_disk(name)
        if self.hparams.checkpoint_ram:
            if mode == 'val' and False:  # DISABLED
                # Check if model checkpointing is working
                if len(self.best_loss[mode]['model']) != 0:
                    keys = list(self.best_loss[mode]['model'].keys())
                    print(self.best_loss[mode]['model'][keys[0]])
                    print(self.state_dict()[keys[0]])
                    print(self.best_loss[mode]['model'][keys[0]] == self.state_dict()[keys[0]])
            self.best_loss[mode]['model'] = copy.deepcopy(self.state_dict())

    def save_checkpoint_to_disk(self, name):
        # Create director for checkpoint
        # This will only work if the program is run from the same location
        directory = './checkpoints/' + self.hparams.experiment + '/' +\
            str(self.logger.version) + '/'
        if os.path.isdir(directory) is False:
            print(f'Making directory at: {directory}')
            os.makedirs(directory)
        self.trainer.save_checkpoint(directory + name)

        # Delete old checkpoint
        if hasattr(self, 'old_checkpoint_path'):
            if self.old_checkpoint_path != directory + name:
                os.remove(self.old_checkpoint_path)
        self.old_checkpoint_path = directory + name

    def train_loss(self, batches=100, loading_bar=False, skip_logging=False):
        """This is only called if logging is enabled
        Many forecasts are mode on the train dataset to calculate a precise average loss
        This loss is logged and also checked if it is the best loss so far
        """
        loss, mae, mse_per_dim, mae_per_dim = self.loss_base(batches, self.dataloaders['train'],
                self.training_step, loading_bar)
        if skip_logging is False:
            self.log('train_loss', loss)
            if loss < self.best_loss['train']['mse']:
                self.best_loss['train']['mse'] = loss
                self.best_loss['train']['mae'] = mae
                self.best_loss['train']['mse_per_dim'] = mse_per_dim
                self.best_loss['train']['mae_per_dim'] = mae_per_dim
                self.log('best/train_loss', loss)
                # No need for this so far
                #self.save_checkpoint(f'{float(loss):.3}' + 'train__loss' + '.ckpt', 'train')
        return loss

    def val_loss(self, batches=100, loading_bar=False, skip_logging=False):
        """This is only called if logging is enabled
        Many forecasts are mode on the val dataset to calculate a precise average loss
        This loss is logged and also checked if it is the best loss so far
        """
        loss, mae, mse_per_dim, mae_per_dim = self.loss_base(batches, self.dataloaders['val'], self.training_step,
                             loading_bar)
        if skip_logging is False:
            self.log('val_loss', loss)
            if loss < self.best_loss['val']['mse']:
                self.best_loss['val']['mse'] = loss
                self.best_loss['val']['mae'] = mae
                self.best_loss['val']['mse_per_dim'] = mse_per_dim
                self.best_loss['val']['mae_per_dim'] = mae_per_dim
                experiement_loss, mae, mse_per_dim, mae_per_dim = self.loss_base(
                        batches, self.dataloaders['test'], self.training_step, loading_bar)
                self.best_loss['experiment']['mse'] = experiement_loss
                self.best_loss['experiment']['mae'] = mae
                self.best_loss['experiment']['mse_per_dim'] = mse_per_dim
                self.best_loss['experiment']['mae_per_dim'] = mae_per_dim
                self.log('best/experiment_loss', experiement_loss)
                self.log('best/val_loss', loss)
                self.save_checkpoint(f'{float(loss):.3}' + 'val__loss' + '.ckpt', 'val')
        return loss

    def test_loss(self, batches=100, loading_bar=False, skip_logging=False):
        """This is only called if logging is enabled
        Many forecasts are mode on the test dataset to calculate a precise average loss
        This loss is logged and also checked if it is the best loss so far
        """
        loss, mae, mse_per_dim, mae_per_dim = self.loss_base(batches, self.dataloaders['test'], self.training_step,
                              loading_bar)
        if skip_logging is False:
            self.log('test_loss', loss)
            if loss < self.best_loss['test']['mse']:
                self.best_loss['test']['mse'] = loss
                self.best_loss['test']['mae'] = mae
                self.best_loss['test']['mse_per_dim'] = mse_per_dim
                self.best_loss['test']['mae_per_dim'] = mae_per_dim
                self.log('best/test_loss', loss)
                self.save_checkpoint(f'{float(loss):.3}' + 'test__loss' + '.ckpt', 'test')
        return loss


class forier_analysis():
    """Inherit this class in the dataloader and call these functions with dataset_dict which has the
    format:
        {ts:'catagorical':
                {catagry: data},
            'exogenial':
                {exo: data},
            'series': data}
    """
    def average_time_series(self, dataset_dict):
        """Plot the average time series
        """
        # Plot 10 days of the average time series
        all_series = []
        for key in dataset_dict:
            all_series.append(dataset_dict[key]['series'])
        all_series = np.asarray(all_series)
        mean = all_series.mean(axis=0)

        days = [7, 10, 50]
        path = '/home/will/Documents/period_transformer/code/stuff/fft_stuff/'
        x = np.arange(len(mean))/96
        for d in days:
            plt.plot(x[:d*96], mean[:d*96])
            plt.grid(visible=True)
            plt.xlabel('Day')
            plt.ylabel('Amplitude')
            plt.title(f'Average time series, {d} days')
            plt.savefig(path+f'average_time_series_{d}_days'+'.pdf', format='pdf', bbox_inches='tight')
            plt.close()
        breakpoint()


    def fft_average_time_series(self, dataset_dict):
        """Calculate the average time series and perform the fft
        """
        # Average time series
        # pylint: disable=E1101
        # Take FFT of each time series and compute average
        y_all = []
        for key in dataset_dict:
            x, y = self.fft(dataset_dict[key]['series'], period_length=96)
            y_all.append(y)
        average_y = np.asarray(y_all).mean(axis=0)

        path = '/home/will/Documents/period_transformer/code/stuff/fft_stuff/'

        # Plot 10 days
        cut_off = sum(x>10)
        plt.plot(x[cut_off:], average_y[cut_off:])
        plt.grid(visible=True)
        plt.xlabel('day')
        plt.ylabel('Amplitude')
        plt.title(f'Average dataset FFT 10 days')
        plt.savefig(path+'average_dataset_fft_10_days'+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        # Plot 1 day
        cut_off = sum(x>1)
        plt.plot(x[cut_off:], average_y[cut_off:])
        plt.grid(visible=True)
        plt.xlabel('day')
        plt.ylabel('Amplitude')
        plt.title(f'Average dataset FFT 1 days')
        plt.savefig(path+'average_dataset_fft_1_days'+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        # Plot entire fft series
        plt.plot(x, average_y)
        plt.grid(visible=True)
        plt.xlabel('day')
        plt.ylabel('Amplitude')
        plt.title(f'Average dataset FFT')
        plt.savefig(path+'average_dataset_fft_all_time'+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        # Plot both weekly and yearly period
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        ax1.plot(x, average_y)
        ax1.grid(visible=True)
        ax2.plot(x, average_y)
        ax2.grid(visible=True)
        ax1.set_xlim(0, 10)
        ax2.set_xlim(200, 600)

        # hide the spines between ax1 and ax2
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        d = .015 # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((1-d,1+d), (-d,+d), **kwargs)
        ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d,+d), (1-d,1+d), **kwargs)
        ax2.plot((-d,+d), (-d,+d), **kwargs)

        f.suptitle(f'FFT weekly and yearly periods')
        f.text(0.5, 0.04, 'Day', ha='center', va='center')
        ax1.set_ylabel('Amplitude')
        plt.savefig(path+f'fft_both'+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        print('FFT of dataset saved')

    def fft_singe_time_series(self, dataset_dict):
        """Save the FFT plot of single time series.  This will be useful to compare to the
        fft_average_time_series() function
        """
        num_plots = 5
        series_length = 105120-1   # 100 days
        keys = np.asarray(list(dataset_dict.keys()))
        rand_indxs = np.random.randint(0, len(keys), (num_plots))
        rand_keys = keys[rand_indxs]

        start_cut = np.random.randint(
            low=0,
            high=len(dataset_dict[rand_keys[0]]['series']) - series_length,
            size=(num_plots))
        end_cut = start_cut + series_length
        rand_series = [
            dataset_dict[k]['series'][start_cut[idx]:end_cut[idx]] for idx, k in enumerate(rand_keys)]

        path = '/home/will/Documents/period_transformer/code/stuff/fft_stuff/single_series/'
        for idx, s in enumerate(rand_series):
            # Plot original series
            plt.plot(np.arange(len(s))/96, s, linewidth=0.1)
            plt.grid(visible=True)
            plt.xlabel('day')
            plt.ylabel('kW')
            plt.title(f'Series {rand_keys[idx]} ({idx}), length {series_length}')
            plt.savefig(path+f'series_{idx}'+'.svg', format='svg', bbox_inches='tight')
            plt.close()

            # Plot whole FFT
            x, y = self.fft(s, period_length=96)
            plt.plot(x, y)
            plt.grid(visible=True)
            plt.xlabel('day')
            plt.ylabel('Amplitude')
            plt.title(f'Whole FFT {rand_keys[idx]} ({idx}), length {series_length}')
            plt.savefig(path+f'whole_fft_{idx}'+'.svg', format='svg', bbox_inches='tight')
            plt.close()

            # Plot FFT
            x, y = self.fft(s, period_length=96)
            cut_off = sum(x>10)
            plt.plot(x[cut_off:], y[cut_off:])
            plt.grid(visible=True)
            plt.xlabel('day')
            plt.ylabel('Amplitude')
            plt.title(f'FFT {rand_keys[idx]} ({idx}), length {series_length}')
            plt.savefig(path+f'fft_{idx}'+'.svg', format='svg', bbox_inches='tight')
            plt.close()

            # Plot both axis
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            x, y = self.fft(s, period_length=96)

            ax1.plot(x, y)
            ax1.grid(visible=True)
            ax2.plot(x, y)
            ax2.grid(visible=True)
            ax1.set_xlim(0, 2)
            ax2.set_xlim(200, 600)

            # hide the spines between ax1 and ax2
            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)

            d = .015 # how big to make the diagonal lines in axes coordinates
            # arguments to pass plot, just so we don't keep repeating them
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d,1+d), (-d,+d), **kwargs)
            ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d,+d), (1-d,1+d), **kwargs)
            ax2.plot((-d,+d), (-d,+d), **kwargs)

            f.suptitle(f'FFT ({idx})')
            f.text(0.5, 0.04, 'Day', ha='center', va='center')
            ax1.set_ylabel('Amplitude')
            plt.savefig(path+f'fft_both_{idx}'+'.svg', format='svg', bbox_inches='tight')
            plt.close()

        print('FFT plots of single time series saved')
        breakpoint()

    def fft(self, series, period_length=1):
        """Perform the fft, returns the frequency domain.  A single step is treated as a time period
        of 1 (as appose to the whole series length havign a length of 1)
        period length: The number of time steps that constitutes a single period
        """
        frq = abs(np.fft.rfft(series))
        x_axis = np.arange(len(frq))
        # Change time period to days instead of whole length of time series
        x_axis = x_axis/len(series)*period_length
        # Convert from frequency to time period
        x_axis = x_axis**-1
        return x_axis, frq

    def fft_test(self):
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        import scipy.fftpack

        # Test fft()
        sf = 1/50  # sampling frequency
        sin = np.sin(np.arange(500)*np.pi*2*sf)
        sf_2 = 1/25
        sin_2 = np.sin(np.arange(500)*np.pi*2*sf_2)

        both = sin+sin_2
        x, y = self.fft(both)
        print('plotting...')
        path = '/home/will/Documents/period_transformer/fourier/'
        plt.plot(x,y)
        plt.savefig(path+'delme2.svg', format='svg', bbox_inches='tight')
        plt.close()


def forecast_fft(model, dataloader, args, num_forecasts):
    """Plot the fft over the loss of the forecast horizon.
        The first plot is zoomed in over 10 days and the second is over the whole horizon.
    """
    x, x_hat, y, stats, forecast_loss = get_forecast(model, dataloader, args, num_forecasts)
    # Get loss of forecast
    loss = F.mse_loss(x_hat, y, reduction='none')
    # Change shape from (b, hirozon_length, 1) to (b, hirizon_length)
    loss = loss[:,:,0]

    # Get fft for each individual forecast and the average
    y_axis_all = []
    for l in tqdm(loss):
        # x_axis is always the same
        x_axis, y_axis = forier_analysis.fft('', l, period_length=96)
        y_axis_all.append(y_axis)
    y_axis_all = np.asarray(y_axis_all)
    mean_fft_loss_y = y_axis_all.mean(axis=0)

    path = '/home/will/Documents/period_transformer/code/stuff/fft_stuff/loss/'

    # FFT of loss, 10 days
    cut_off = sum(x_axis>10)
    plt.plot(x_axis[cut_off:], mean_fft_loss_y[cut_off:])
    plt.grid(visible=True)
    plt.xlabel('day')
    plt.ylabel('Amplitude')
    plt.title(f'Loss FFT 10 days')
    plt.savefig(path+'10_day_loss_fft'+'.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # FFT of loss, entire length
    plt.plot(x_axis, mean_fft_loss_y)
    plt.grid(visible=True)
    plt.xlabel('day')
    plt.ylabel('Amplitude')
    plt.title(f'Loss FFT')
    plt.savefig(path+'loss_fft'+'.pdf', format='pdf', bbox_inches='tight')
    plt.close()


    print('Plots saved')
