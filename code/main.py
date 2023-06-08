"""This is the main python file.
"""

import matplotlib
import argparse
import torch
import sys
import dataloader
from models import MyEarlyStopping
from models.vanilla_transformer import VanillaTransformer
from models.autoformer import Autoformer
from models.fedformer import FEDformer
from models.informer import Informer
import pytorch_lightning as pl
import evaluate_model
import warnings
import os
import pickle
import pandas as pd
import copy




def main():
    warnings.filterwarnings(
        "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
    )
    warnings.filterwarnings(
        "ignore", ".*You defined a `validation_step` but have no `val_dataloader`*"
    )
    parser = argparse.ArgumentParser(description='Seasonal Transformer Project')
    # PROGRAM level args
    parser.add_argument('--log', action='store_true',
                        help='Enable logging',)
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Set the batch size',)
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Set the learning rate',)
    parser.add_argument('--cpu', action='store_true',
                        help='Train on CPU instead of GPU',)
    parser.add_argument('--experiment', default='default',
                        help='The name of the experiment for logging',)
    parser.add_argument('--end', type=int, default=0,
                        help='View a prediction at the end of training',)
    parser.add_argument('--validate_every', type=int, default=500,
                        help='How often to run the validation set')
    parser.add_argument('--horizon_length', type=int,
                        help='Number of time steps to predict.')
    parser.add_argument('--history_length', type=int,
                        help='history_length: The number of historical steps fed into the '+\
                        'time series model')
    parser.add_argument('--continue_training', type=str, default=None,
                        help='Set to path of checkpoint to continue training')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Set to path of checkpoint to load the model.  The learning rate '+\
                        'will have also reset so if training continues, loss might initially increase.')
    parser.add_argument('--patience', type=int, default=None,
                        help='The number of validantion steps with no improvement before early'+\
                        'stopping.  Validation frequency is set with --validate_every')
    parser.add_argument('--dataset', type=str, default=None,
                        help='The function name of the dataset.')
    parser.add_argument('--experiment_batch_name', type=str, default=None,
                        help='The name of the experiment for results pickling')
    parser.add_argument('--validate_batches', type=int, default=100,
                        help='The number of batches to use to calculate evaluation stats during' +\
                        'training.  If --log_every_n_steps is <= 5 validate_batches is set to 5')
    parser.add_argument('--checkpoint', action='store_true', default=False,
                        help='Enable checkpointing, writes a lot to the disk!',)
    parser.add_argument('--checkpoint_ram', action='store_true', default=True,
                        help='Save best experiment checkpoint in the RAM',)
    parser.add_argument('--merge_splits', action='store_true', default=False,
                        help='Merge train, val, test splits in the same manor that the Autoformer' +\
                        'and the Informer do')
    parser.add_argument('--univariate', action='store_true', default=False,
                        help='')
    parser.add_argument('--single_series', action='store_true', default=False,
                        help='')
    parser.add_argument('--split_ratios', nargs='+',  type=float, default=None,
                        help='The train, val and test split ratios')
    parser.add_argument('--no_shuffle', action='store_true', default=False,
                        help='Dont shuffle the dataloader',)
    parser.add_argument('--crop_data', default=0, type=float,
                        help='Percentage of the data to keep.  Data is removed in the time axis',)
    parser.add_argument('--keep_dims', default=0, type=int,
                        help='The number of dims to keep of the dataset.  This '+\
                        'is used for a specific experiment and is disabled when equal 0'),
    parser.add_argument('--disable_progress_bar', action='store_true', default=False)
    parser.add_argument('--input_differentiation', action='store_true', default=False)
    parser.add_argument('--univariate_multi_model', type=int, default=-1)
    parser.add_argument('--embedding', type=str, default='BasicEmbedding')
    parser.add_argument('--dimensions_per_kernal', type=int, default=0)
    parser.add_argument('--independence', type=float, default=0, help='0->1.  0 is off')


    parser = pl.Trainer.add_argparse_args(parser)
    # Figure out which model to use
    parser.add_argument("--model_name", type=str, default="SeasonalTransformer", help="SeasonalTransformer or Enhancing")

    # This line is key to pull the model name
    temp_args, _ = parser.parse_known_args()
    # Add model specific args
    globals()[temp_args.model_name].add_model_specific_args(parser) 

    args = parser.parse_args()
    assert not (args.univariate == True and args.univariate_multi_model > -1)

    args.device = 'cpu' if args.cpu else 'cuda'
    if args.max_epochs == None:
        args.max_epochs = -1
    print(args.experiment)
    # --------------------Datasets and dataloaders--------------------
    dataset_args = {
        'history_length': args.history_length,
        'horizon_length':args.horizon_length,
        'device': args.device,
        'merge_splits': args.merge_splits,
        'univariate': args.univariate,
        'keep_dims': args.keep_dims,
        'univariate_multi_model': args.univariate_multi_model,
        'args': args}
    if args.split_ratios is not None:
        split = {'train': args.split_ratios[0],
                 'val':  args.split_ratios[1],
                 'test': args.split_ratios[2]}
        dataset_args['split'] = split

    datasets = getattr(dataloader, args.dataset)(**dataset_args)
    for key in datasets:
        pass
        #print(f'len(datasets["{key}"]): {len(datasets[key])}')

    if args.cpu:
        persistent_workers=False
        num_workers=0
        print('Setting workers to 0')
    else:
        persistent_workers=True
        num_workers=6
    dataloaders = {}
    for key in datasets:
        dataloaders[key] = torch.utils.data.DataLoader(
            dataset=datasets[key],
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=persistent_workers,
            num_workers=num_workers)

    # Create instance of model from model_name
    if args.load_model is not None:
        model = globals()[args.model_name].load_from_checkpoint(
            args.load_model,
            hparams=args,
            dataloaders=dataloaders,
            learning_rate=args.learning_rate)
        breakpoint()  # If we start training the optimizer might kick the weights about
        # If you continue the model will either continue where it left off (if continue_training is
        # set). Otherwise it will start a new training instance and start training with the
        # recovered weights
    else:
        model = globals()[args.model_name](args, dataloaders, learning_rate=args.learning_rate)

    if args.log:
        logger = pl.loggers.TensorBoardLogger(
            save_dir='logs/tensorboard',
            name=args.experiment,
            default_hp_metric=False)
    else:
        logger = False
    if args.patience is None:
        callbacks = []
    else:
        callbacks=[MyEarlyStopping(monitor="val_loss",
                                   patience=args.patience,
                                   verbose=False,
                                   check_on_train_epoch_end=False)]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        accelerator='cpu' if args.cpu else 'gpu',
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=False,
        enable_progress_bar=not args.disable_progress_bar)
    print('Training starting...')

    if args.continue_training is not None:
        trainer.fit(model, dataloaders['training_loop'], ckpt_path=args.continue_training)
    else:
        trainer.fit(model, dataloaders['training_loop'])
    print('training complete')


    # Restore the best val model
    model.load_state_dict(model.best_loss['val']['model'])
    model.to(args.device)


    if args.log and args.experiment_batch_name is not None:
        directory = './results/'
        if os.path.isdir(directory) is True:
            # Update the experiment_batch_name to include the version
            model.hparams['experiment_batch_name'] += f'_{logger.version}'
            df = pd.DataFrame({
                'experiment_mse': model.best_loss['experiment']['mse'].item(),
                'experiment_mae': model.best_loss['experiment']['mae'].item(),
                'best_experiment_mse_per_dim': [model.best_loss['experiment']['mse_per_dim']],
                'best_experiment_mae_per_dim': [model.best_loss['experiment']['mae_per_dim']],
                'best_train_mse': model.best_loss['train']['mse'].item(),
                'best_train_mae': model.best_loss['train']['mae'].item(),
                'best_train_mse_per_dim': [model.best_loss['train']['mse_per_dim']],
                'best_train_mae_per_dim': [model.best_loss['train']['mae_per_dim']],
                'best_val_mse': model.best_loss['val']['mse'].item(),
                'best_val_mae': model.best_loss['val']['mae'].item(),
                'best_val_mse_per_dim': [model.best_loss['val']['mse_per_dim']],
                'best_val_mae_per_dim': [model.best_loss['val']['mae_per_dim']],
                'best_test_mse': model.best_loss['test']['mse'].item(),
                'best_test_mae': model.best_loss['test']['mae'].item(),
                'best_test_mse_per_dim': [model.best_loss['test']['mse_per_dim']],
                'best_test_mae_per_dim': [model.best_loss['test']['mae_per_dim']],
                'hparams': [model.hparams],
                'experiment_name': model.hparams.experiment,
                'experiment_batch_name': model.hparams.experiment_batch_name,
                'version': logger.version,
                'history': model.hparams.history_length,
                'horizon': model.hparams.horizon_length,
                'keep_dims': model.hparams.keep_dims,
                'univariate_multi_model': model.hparams.univariate_multi_model,
                'independence': model.hparams.independence,
                'dataset': model.hparams.dataset})
            if args.input_differentiation:
                df['input_differentiation_val'] = [model.input_differentiation('val')]
                df['input_differentiation_test'] = [model.input_differentiation('test')]
            file_name = args.experiment_batch_name + '_' + str(logger.version) + '.pickle'
            if os.path.isfile(directory+file_name):
                previous_df = pickle.load(open(directory + file_name, 'rb'))
                df = pd.concat((previous_df, df), ignore_index=True)
            else:
                print('New results pickle...')
            pickle.dump(df, open(directory + file_name, 'wb'))
            print('\n---Results---')
            print('Loss:')
            print(f"  mse: {model.best_loss['experiment']['mse']}")
            print(f"  mae: {model.best_loss['experiment']['mae']}\n")

        else:
            print('No results directory for results')

    if args.end and args.log:
        while True:
            evaluate_model.forecast_fft(model, dataloaders['test'], args, num_forecasts=10000)
            evaluate_model.forecast(model, dataloaders['test'], args)
            breakpoint()


def find_and_plot_lr(trainer, model, dataloaders):
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model, dataloaders['training_loop'])
    # Results can be found in
    lr_finder.results

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    return


if __name__ == '__main__':
    main()
