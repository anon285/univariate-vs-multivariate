import pandas as pd
import numpy as np
import pickle

def main():
    """Convert csv file into the following format:
        {
            'data':{
                <Time_series_keys>{          - One key/dictionary for each series
                    'series': np.array(),    - One dimension for each series variable
                    'mean': np.array(),      - Mean of each series varibale (for unnormalisation)
                    'std': np.array(),}}     - STD of each series variable (for unnormalisation)
            'description': np.array()        - Describes each dimension of the series
        }
        Download csv file from:
            https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/?p=%2F&mode=list
            https://github.com/thuml/Autoformer
        Sampled: Daily
        Features: - 8
        Date: 1990 - 2010  # In the paper they say it's from 1990 - 2016?
        Length: 7588
    """

    csv = pd.read_csv('exchange_rate.csv')
    breakpoint()

    dataset = {'data': {},
               'description': np.array([])}
    series_name = 'exchange_rate'
    # Drop date column
    csv.drop(columns='date', inplace=True)

    # Convert from float 64 to 32
    csv = csv.astype(np.float32)


    # Save mean and std
    dataset['data'][series_name] = {}
    dataset['data'][series_name]['mean'] = csv.mean().to_numpy()
    dataset['data'][series_name]['std'] = csv.std().to_numpy()

    # Normalise
    dataset['data'][series_name]['series'] = \
        (csv - dataset['data'][series_name]['mean']) / dataset['data'][series_name]['std']
    print('The following should be close to 0')
    print(dataset['data'][series_name]['series'].mean())
    print('The following should be close to 1')
    print(dataset['data'][series_name]['series'].std())

    dataset['description'] = csv.columns.to_numpy()

    convert_to_numpy = False  #  This would save RAM
    if convert_to_numpy:
        dataset['data'][series_name]['series'] = dataset['data'][series_name]['series'].to_numpy()


    pickle.dump(dataset, open(f'{series_name}_dictionary.pickle', 'wb'))
    breakpoint()



if __name__ == '__main__':
    main()


