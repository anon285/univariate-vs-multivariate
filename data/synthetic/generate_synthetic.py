import pickle
import numpy as np
import pandas as pd


dataset_name = 'synthetic'
dataset_dict_path = '../autoformer/electricity/electricity_dictionary.pickle'
electricity = pickle.load(open(dataset_dict_path, 'rb'))
raw_electricity= pickle.load(open(dataset_dict_path, 'rb'))
electricity = electricity['data']['electricity']['series']['0']
electricity.name = 'electricity'

dataset_dict_path = '../autoformer/traffic/traffic_dictionary.pickle'
traffic = pickle.load(open(dataset_dict_path, 'rb'))
raw_traffic = pickle.load(open(dataset_dict_path, 'rb'))
traffic = traffic['data']['traffic']['series']['0']
traffic.name = 'traffic'

dataset_dict_path = '../autoformer/weather/weather_dictionary.pickle'
weather = pickle.load(open(dataset_dict_path, 'rb'))
raw_weather = pickle.load(open(dataset_dict_path, 'rb'))
weather = weather['data']['weather']['series']['p (mbar)']
weather.name = 'weather'

dataset_dict_path = '../autoformer/ETT-small/m2_dictionary.pickle'
ett = pickle.load(open(dataset_dict_path, 'rb'))
raw_ett = pickle.load(open(dataset_dict_path, 'rb'))
ett = ett['data']['m2']['series']['HUFL']
ett.name = 'ett'

# Now that we have the series, crop them down to size
df = pd.concat((electricity, traffic, weather, ett), axis=1)
df = df[0:len(traffic)]  # Traffic is the shortest dataset at 17544

# Get mean and std
mean = []
mean.append(raw_electricity['data']['electricity']['mean'][0])
mean.append(raw_traffic['data']['traffic']['mean'][0])
mean.append(raw_weather['data']['weather']['mean'][0])
mean.append(raw_ett['data']['m2']['mean'][0])
std = []
std.append(raw_electricity['data']['electricity']['std'][0])
std.append(raw_traffic['data']['traffic']['std'][0])
std.append(raw_weather['data']['weather']['std'][0])
std.append(raw_ett['data']['m2']['std'][0])

tmp_series = {'mean': mean, 'std': std, 'series': df}

dataset = {'data': {dataset_name:tmp_series},
           'description': df.columns.values.tolist()}

pickle.dump(dataset, open(f'{dataset_name}_dictionary.pickle', 'wb'))
