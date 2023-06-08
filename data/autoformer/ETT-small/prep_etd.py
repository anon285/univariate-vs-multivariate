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
        Sampled: Hourly, 15 minutes
        Features: 7
        Date: 2016 - 2018

        h1 length: 17420
        h2 length: 17420
        m1 length: 69680
        m2 length: 69680
    """

    csv = {}
    csv['h1'] = pd.read_csv('ETTh1.csv')
    csv['h2'] = pd.read_csv('ETTh2.csv')
    csv['m1'] = pd.read_csv('ETTm1.csv')
    csv['m2'] = pd.read_csv('ETTm2.csv')

    # Loop through each csv, creating a separate dataset for each one
    for key in csv:
        dataset = {'data': {},
                   'description': np.array([])}
        # Drop date column
        csv[key].drop(columns='date', inplace=True)

        # Convert from float 64 to 32
        csv[key] = csv[key].astype(np.float32)

        # Save mean and std
        dataset['data'][key] = {}
        dataset['data'][key]['mean'] = csv[key].mean().to_numpy()
        dataset['data'][key]['std'] = csv[key].std().to_numpy()

        # Normalise
        dataset['data'][key]['series'] = \
            (csv[key] - dataset['data'][key]['mean']) / dataset['data'][key]['std']

        dataset['description'] = csv[key].columns.to_numpy()


        pickle.dump(dataset, open(f'{key}_dictionary.pickle', 'wb'))



if __name__ == '__main__':
    main()
