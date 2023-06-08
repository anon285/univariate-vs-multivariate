"""This new dataloader uses the new format:

{
    'data':{
        <Time_series_keys>{          - One key/dictionary for each series
            'series': np.array(),    - One dimension for each series variable
            'mean': np.array(),      - Mean of each series varibale (for unnormalisation)
            'std': np.array(),}}     - STD of each series variable (for unnormalisation)
    'description': np.array()        - Describes each dimension of the series

    __getitem__() should return:
        x - the time series
        y - the forecast horizon
        stats - {'mean': _:, 'std': _:, 'offset': _:, 'name': _}
        the time series, mean and std
}
"""
import pickle
import numpy as np
import torch
import copy
import random
from autoformer_dataloader import *
from fedformer_dataloader import *
from informer_dataloader import *


def process(dataset_dict, keep_dims, univariate_multi_model, args):
    if keep_dims != 0:
        drop_dims(dataset_dict, keep_dims, args)


def drop_dims(dataset_dict, keep_dims, args):
    """args:
        dataset_dict: our time series
        dims: the number of dimensions that are to be kept
    """
    print('Dropping dims')
    series_name = list(dataset_dict['data'].keys())[0]
    is_random = True
    if is_random:
        num_dims = dataset_dict['data'][series_name]['series'].shape[-1]
        random_dims = random.sample(range(0, num_dims), keep_dims)
    else:
        random_dims = range(0, keep_dims)

    args.keep_dims_selection = random_dims

    dataset_dict['data'][series_name]['series'] = \
            dataset_dict['data'][series_name]['series'].iloc[:, random_dims]

    return dataset_dict


def autoformer_electricity(history_length, horizon_length, device, keep_dims, merge_splits,
        univariate, univariate_multi_model, split={'train': 0.7, 'val': 0.1, 'test': 0.2},
        single_series=False, args=None):
    """Loads the df from the Autoformer paper.  A data split from the paper of 7:1:2
        Sampled: Hourly
        Features: 321
        Date: 2016 - 2019  # In the paper they say it's from 2012 - 2014?
        Length: 26304
    """
    dataset_dict_path = '../data/autoformer/electricity/electricity_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    if univariate and single_series:
        # Remove all the series apart from OT
        dataset_dict['data']['electricity']['series'] = \
            dataset_dict['data']['electricity']['series']['OT']

    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


def autoformer_exchange_rate(history_length, horizon_length, device, keep_dims, merge_splits,
        univariate, univariate_multi_model, split={'train': 0.7, 'val': 0.1, 'test': 0.2},
        args=None):
    """Loads the df from the Autoformer paper.  A data split from the paper of 7:1:2
        Sampled: Daily
        Features: - 8
        Date: 1990 - 2010  # In the paper they say it's from 1990 - 2016?
        Length: 7588
    """
    dataset_dict_path = '../data/autoformer/exchange_rate/exchange_rate_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


def autoformer_traffic(history_length, horizon_length, device, keep_dims, merge_splits, univariate,
                       univariate_multi_model,
        split={'train': 0.7, 'val': 0.1, 'test': 0.2}, args=None):
    """Loads the df from the Autoformer paper.  A data split from the paper of 7:1:2
        Sampled: Hourly
        Features: 862
        Date: - 2016 - 2018
        Length: 17544
    """
    dataset_dict_path = '../data/autoformer/traffic/traffic_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


def autoformer_weather(history_length, horizon_length, device, keep_dims, merge_splits, univariate,
                       univariate_multi_model,
        split={'train': 0.7, 'val': 0.1, 'test': 0.2}, args=None):
    """Loads the df from the Autoformer paper.  A data split from the paper of 7:1:2
        Sampled: 10 minutes
        Features: 21
        Date: 2020
        Length: 52696
    """
    dataset_dict_path = '../data/autoformer/weather/weather_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


def autoformer_illness(history_length, horizon_length, device, keep_dims, merge_splits, univariate,
                       univariate_multi_model,
        split={'train': 0.7, 'val': 0.1, 'test': 0.2}, args=None):
    """Loads the df from the Autoformer paper.  A data split from the paper of 7:1:2
        Sampled: Weekly
        Features: 7
        Date: 2002 - 2021
        Length: 966
    """
    dataset_dict_path = '../data/autoformer/illness/illness_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


def autoformer_ETD_h1(history_length, horizon_length, device, keep_dims, merge_splits, univariate,
                      univariate_multi_model,
        split={'train': 0.6, 'val': 0.2, 'test': 0.2}, args=None):
    """Loads the ETT df from the Autoformer paper.  A data split from the paper of 6:2:2
        Sampled: Hourly
        Features: 7
        Date: 2016 - 2018
        Length: 17420
    """
    dataset_dict_path = '../data/autoformer/ETT-small/h1_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


def autoformer_ETD_h2(history_length, horizon_length, device, keep_dims, merge_splits, univariate,
                      univariate_multi_model,
        split={'train': 0.6, 'val': 0.2, 'test': 0.2}, args=None):
    """Loads the ETT df from the Autoformer paper.  A data split from the paper of 6:2:2
        Sampled: Hourly
        Features: 7
        Date: 2016 - 2018
        Length: 17420
    """
    dataset_dict_path = '../data/autoformer/ETT-small/h2_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


def autoformer_ETD_m1(history_length, horizon_length, device, keep_dims, merge_splits, univariate,
                      univariate_multi_model,
        split={'train': 0.6, 'val': 0.2, 'test': 0.2}, args=None):
    """Loads the ETT df from the Autoformer paper.  A data split from the paper of 6:2:2
        Sampled: 15 minutes
        Features: 7
        Date: 2016 - 2018
        Length: 69680
    """
    dataset_dict_path = '../data/autoformer/ETT-small/m1_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


def autoformer_ETD_m2(history_length, horizon_length, device, keep_dims, merge_splits, univariate,
                      univariate_multi_model,
        split={'train': 0.6, 'val': 0.2, 'test': 0.2}, args=None):
    """Loads the ETT df from the Autoformer paper.  A data split from the paper of 6:2:2
        Sampled: 15 minutes
        Features: 7
        Date: 2016 - 2018
        Length: 69680
    """
    dataset_dict_path = '../data/autoformer/ETT-small/m2_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


def synthetic(history_length, horizon_length, device, keep_dims, merge_splits,
        univariate, univariate_multi_model, split={'train': 0.7, 'val': 0.1, 'test': 0.2},
        args=None):
    """Loads the independent dimensions dataset df, comprised of the first dimension from the
    electricity, traffic, weather and ett datasets.
    A data split from the paper of 12/4/4 months is used.
        Features: 4
        Length: 17544
    """
    dataset_dict_path = '../data/synthetic/synthetic_dictionary.pickle'
    dataset_dict = pickle.load(open(dataset_dict_path, 'rb'))
    process(dataset_dict, keep_dims, univariate_multi_model, args)
    return dataset_dict_split(
        dataset_dict, history_length, horizon_length, device, split=split,
        merge_splits=merge_splits, univariate=univariate,
        univariate_multi_model=univariate_multi_model,
        independence=args.independence, crop_data=args.crop_data)


# ----------------------------------------

def dataset_dict_split(dataset_dict,  history_length, horizon_length, device, split, merge_splits,
                       univariate, univariate_multi_model, independence, crop_data=0):
    """This function takes the dataset dict and splits them into train, val and test.
    """
    series_length = len(list(dataset_dict['data'].values())[0]['series'])
    num_sequences_per_series = series_length - (history_length + horizon_length - 1)

    if merge_splits:
        m = history_length
    else:
        m = 0

    if crop_data != 0:
        print(crop_data)

    offset = {'train': [], 'val': [], 'test': []}
    offset['train'] = [int(series_length*split['train'] * crop_data),
                       int(series_length*split['train'])]
    offset['val'] = [offset['train'][1] - m,
                     int(series_length*(split['train']+split['val']))]
    offset['test'] = [offset['val'][1] - m,
                      series_length]

    # Overwrite each series in the following loop
    dataset_split = {'train': copy.deepcopy(dataset_dict),
                     'val': copy.deepcopy(dataset_dict),
                     'test': copy.deepcopy(dataset_dict)}
    for s in dataset_dict['data']:
        for key in offset:
            dataset_split[key]['data'][s]['series'] = dataset_dict['data'][s]['series']\
            [offset[key][0]: offset[key][1]]
            #print(f'dataset_split[{key}].shape: {dataset_split[key]["data"][s]["series"].shape}')
    datasets = {}
    for key in dataset_split:
        datasets[key] = ToPytorchDataset(dataset_split[key], history_length, horizon_length,
                                         univariate, univariate_multi_model, independence, device)
    datasets['training_loop'] = copy.deepcopy(datasets['train'])
    return datasets


class ToPytorchDataset(torch.utils.data.Dataset):
    """Takes a dataset_dict and returns the dataloader
    """
    def __init__(self, dataset_dict, history_length, horizon_length, univariate, 
                 univariate_multi_model, independence, device):
        self.dataset_dict=dataset_dict
        self.series_length = len(list(dataset_dict['data'].values())[0]['series'])
        self.num_sequences_per_series = self.series_length - (history_length + horizon_length - 1)
        self.num_series = len(dataset_dict['data'])
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.univariate = univariate
        self.univariate_multi_model = univariate_multi_model
        self.independence = independence
        self.device = device

        # Convert pandas df to numpy array
        # shape of each series: (length, data_dim)
        for series_name in self.dataset_dict['data']:
            self.dataset_dict['data'][series_name]['series'] = \
                self.dataset_dict['data'][series_name]['series'].to_numpy()
            # If we only have one sequence add a singleton.  We want to keep the sequence dimension,
            # even if we only have one
            if len(self.dataset_dict['data'][series_name]['series'].shape) == 1:
                self.dataset_dict['data'][series_name]['series'] = np.expand_dims(
                    self.dataset_dict['data'][series_name]['series'], axis=1)
        self.data_dim = list(dataset_dict['data'].values())[0]['series'].shape[-1]
        if self.__len__() <= 0:
            shape = list(self.dataset_dict['data'].values())[0]['series'].shape
            raise ValueError(f'Not enough data for dataloader\n  len(self): {self.__len__()}'
                             f'\n  data.shape: {shape}')


    def __len__(self):
        # self.num_series is currently always one.  The traffic dataset counts as a single series
        # A sequence is the a sub section of a series, history+horizon long
        if self.univariate:
            return self.num_sequences_per_series * self.num_series * self.data_dim
        else:
            return self.num_sequences_per_series * self.num_series

    def __getitem__(self, idx):
        if self.univariate_multi_model > -1:
            # Only one dimension will ever be used out of self.data_dim
            univariate_idx = self.univariate_multi_model
            univariate_idx_offset = univariate_idx + 1
            # the length of the dataset will be equal to the multivariate case
        elif self.univariate:
            # Select just one dimension (univariate_idx)
            univariate_idx = int(idx/self.num_sequences_per_series)
            univariate_idx_offset = univariate_idx + 1
        else:
            # Multivariate - select all dimensions
            univariate_idx = 0
            univariate_idx_offset = self.data_dim

        old_series_idx = 0
        history_offset = [idx % self.num_sequences_per_series,
                          idx % self.num_sequences_per_series + self.history_length]
        horizon_offset = [history_offset[1],
                          history_offset[1] + self.horizon_length]

        time_series = list(self.dataset_dict['data'].values())[old_series_idx ]['series']

        if self.independence > 0:
            # Loop through dimensions
            tmp_y = []
            for dim in range(self.data_dim):
                # If this is univariate then we need to get all the other dimensions and add them to 
                selected_dimension = time_series[
                    horizon_offset[0]: horizon_offset[1], dim:dim+1]
                other_dimensions = [i for i in range(self.data_dim) if i != dim]
                other_dimensions = time_series[
                    horizon_offset[0]: horizon_offset[1], other_dimensions]
                # Scale before adding
                selected_dimension = selected_dimension * (1-self.independence)
                other_dimensions = other_dimensions * self.independence

                # Add and merge all dimensions together
                other_dimensions = other_dimensions.sum(axis=1,keepdims=True)
                tmp_y.append(selected_dimension + other_dimensions)
            tmp_y = np.asarray(tmp_y)
            tmp_y = np.squeeze(tmp_y)
            tmp_y = np.transpose(tmp_y, (1, 0))  # shape: (horizon, input_dims)

            y = tmp_y[:, univariate_idx:univariate_idx_offset]
        else:
            y = time_series[
                horizon_offset[0]: horizon_offset[1], univariate_idx:univariate_idx_offset]

        x = time_series[
            history_offset[0]: history_offset[1], univariate_idx:univariate_idx_offset]

        x = torch.tensor(x)
        y = torch.tensor(y)

        stats = {'offset': [history_offset[0], horizon_offset[1]],
                 'name': list(self.dataset_dict['data'].keys())[old_series_idx ],
                 'mean': list(self.dataset_dict['data'].values())[old_series_idx ]['mean'],
                 'std': list(self.dataset_dict['data'].values())[old_series_idx ]['std']}

        return x, y, stats
