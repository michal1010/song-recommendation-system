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

divided_songs = {}
genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']

for genre in genres_to_filter:
    data_raw = pd.read_json(f'annotaded_tracks/{genre}_annotated.json', orient='records', lines=True)
    data_raw = data_raw.drop(["id", 'name', 'explicit', 'id_artist', 'key', 'release_date', 'popularity' ], axis=1)
    data_raw.loc[:, :] = data_raw.loc[:, :] .astype(float)

    train_data = torch.from_numpy(data_raw.values[:, :-1]).float()
    train_targets = torch.from_numpy(data_raw.values[:, -1]).float()
    train_dataset = data.TensorDataset(train_data, train_targets)
    data_loader_train = data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    device = torch.device('cpu')
    model = RegresionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    iters = []
    losses_train = []
    losses_valid = []

    for n in range(550):
        epoch_losses = []
        for x, labels in data_loader_train:
            x, labels = x.to(device), labels.to(device)
            model.train()
            out = model(x)
            out = out.squeeze(dim=1)
            loss = criterion(out, labels)
            optimizer.zero_grad()

            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()
        loss_mean_train = np.array(epoch_losses).mean()
        iters.append(n)
        losses_train.append(loss_mean_train)
    torch.save(model.state_dict(), f'state_dicts/{genre}_model_dict.pt')


