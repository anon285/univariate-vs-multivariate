import copy


def autoformer_dataloader_weather(*args, **kwargs):
    kwargs['args'].root_path = "../data/autoformer/weather/"
    kwargs['args'].data_path = "weather.csv"
    kwargs['args'].seq_len = kwargs['args'].history_length
    kwargs['args'].pred_len = kwargs['args'].horizon_length
    return autoformer_model(*args, **kwargs)

def autoformer_dataloader_traffic(*args, **kwargs):
    kwargs['args'].root_path = "../data/autoformer/traffic/"
    kwargs['args'].data_path = "traffic.csv"
    kwargs['args'].seq_len = kwargs['args'].history_length
    kwargs['args'].pred_len = kwargs['args'].horizon_length
    return autoformer_model(*args, **kwargs)

def autoformer_dataloader_illness(*args, **kwargs):
    kwargs['args'].root_path = "../data/autoformer/illness/"
    kwargs['args'].data_path = "national_illness.csv"
    kwargs['args'].seq_len = kwargs['args'].history_length
    kwargs['args'].pred_len = kwargs['args'].horizon_length
    return autoformer_model(*args, **kwargs)

def autoformer_dataloader_electricity(*args, **kwargs):
    kwargs['args'].root_path = "../data/autoformer/electricity/"
    kwargs['args'].data_path = "electricity.csv"
    kwargs['args'].seq_len = kwargs['args'].history_length
    kwargs['args'].pred_len = kwargs['args'].horizon_length
    return autoformer_model(*args, **kwargs)

def autoformer_dataloader_exchange(*args, **kwargs):
    kwargs['args'].root_path = "../data/autoformer/exchange_rate/"
    kwargs['args'].data_path = "exchange_rate.csv"
    kwargs['args'].seq_len = kwargs['args'].history_length
    kwargs['args'].pred_len = kwargs['args'].horizon_length
    return autoformer_model(*args, **kwargs)

def autoformer_dataloader_ettm2(*args, **kwargs):
    kwargs['args'].root_path = "../data/autoformer/ETT-small/"
    kwargs['args'].data_path = "ETTm2.csv"
    kwargs['args'].seq_len = kwargs['args'].history_length
    kwargs['args'].pred_len = kwargs['args'].horizon_length
    return autoformer_model(*args, **kwargs)


def process(keep_dims, univariate_multi_model):
    if keep_dims != 0:
        print('Keep_dims not yet implemented')
        assert keep_dims == 0
    if univariate_multi_model != -1:
        print('Univariate_multi_model not yet implemented')
        assert univariate_multi_model == 0

def autoformer_model(history_length, horizon_length, device, args, keep_dims,
        univariate_multi_model, merge_splits, univariate,
        split={'train': 0.7, 'val': 0.1, 'test': 0.2}):
    # get a dictionary of all the different datasets
    # lets just get it working with electricity
    import pickle
    #auto_args, flag = pickle.load(open('models/other/Autoformer/my_args/train_args.pickle','rb'))
    #auto_args.root_path = '../data/autoformer/electricity'

    from models.other.Autoformer.data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,\
            Dataset_Pred
    process(keep_dims, univariate_multi_model)
    if keep_dims != 0:
        print('Keep_dims not yet implemented')
        assert keep_dims == 0
    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'custom': Dataset_Custom}

    datasets = {}
    for flag in ['train', 'val', 'test']:
        Data = data_dict[args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':  # I probably won't be using this
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.freq
            Data = Dataset_Pred
        else:  # For train and val
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            univariate=univariate,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq)
        datasets[flag] = data_set
    datasets['training_loop'] = copy.deepcopy(datasets['train'])

    return datasets

