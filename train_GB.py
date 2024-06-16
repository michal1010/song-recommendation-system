import torch, torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import pandas as pd
import torch.utils.data as data
import seaborn as sns
import matplotlib.pyplot as plt
from RegresionModel import RegresionModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

divided_songs = {}
genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']

for genre in genres_to_filter:
    data_raw = pd.read_json(f'annotaded_tracks/{genre}_annotated.json', orient='records', lines=True)
    data_raw = data_raw.drop(["id", 'name', 'explicit', 'id_artist', 'key', 'release_date', 'popularity', \
                            'loudness', 'acousticness', 'explicit', 'valence', 'duration_ms', 'instrumentalness'], axis=1)
    scaler = StandardScaler()
    data_raw.iloc[:, :-1] = scaler.fit_transform(data_raw.iloc[:, :-1])
    data_raw.loc[:, :] = data_raw.loc[:, :] .astype(float)

    train_data = torch.from_numpy(data_raw.values[:, :-1]).float()
    train_targets = torch.from_numpy(data_raw.values[:, -1]).float()

    model_grad = GradientBoostingRegressor()
    model_grad.fit(train_data, train_targets)

    joblib.dump(model_grad, f'state_dicts_GB/{genre}_state_dict.joblib')
