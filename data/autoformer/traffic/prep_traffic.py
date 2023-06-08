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
        Sampled: Hourly
        Features: 862
        Date: - 2016 - 2018
        Length: 17544
    """

    csv = pd.read_csv('traffic.csv')

    dataset = {'data': {},
               'description': np.array([])}
    series_name = 'traffic'
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

    dataset['description'] = csv.columns.to_numpy()

    convert_to_numpy = False  #  This would save RAM
    if convert_to_numpy:
        dataset['data'][series_name]['series'] = dataset['data'][series_name]['series'].to_numpy()


    pickle.dump(dataset, open(f'{series_name}_dictionary.pickle', 'wb'))



if __name__ == '__main__':
    main()

