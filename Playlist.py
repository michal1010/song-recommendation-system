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
import joblib
from NN import RegresionModel

class Playlist:

    def __init__(self, models_parameters_NN: dict, models_parameters_GB) -> None:
        self.number_of_attributes = 5
        self.models_NN = self._create_models_NN(models_parameters_NN)
        self.models_GB = self._create_models_GB(models_parameters_GB)
        self._genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']

    def _create_models_NN(self, models_parameters: dict):
        models = {}
        for genre, parameters_path in models_parameters.items():
            model = RegresionModel(self.number_of_attributes)
            model.load_state_dict(torch.load(parameters_path))
            model.eval()
            models[genre] = model
        return models

    def _create_models_GB(self, models_parameters: dict):
        models = {}
        for genre, parameters_path in models_parameters.items():
            model = joblib.load(parameters_path)
            models[genre] = model
        return models

    def _create_data_loader(self, tracks):
        tracks_tensor = torch.tensor(tracks.values, dtype=torch.float32)
        return data.DataLoader(tracks_tensor, batch_size=self.BATCH_SIZE)

    def create_playlist(self, genre, length, tracks_path, artists_path, model='NN'):
        if genre not in self.models_NN.keys():
            return 'Wrong genre selected choose the one from : pop, trap' \
                    + 'rap, rock, metal, hip hop'
        self._tracks = self._get_tracks(tracks_path, artists_path, genre)

        tracks = self._tracks.drop(["id", 'name', 'explicit', 'id_artist', 'key', 'release_date', 'popularity', \
                                    'loudness', 'acousticness', 'explicit', 'valence',
                                    'duration_ms', 'instrumentalness'], axis=1).to_numpy().astype(np.float32)
        rating = []
        if model == 'NN':
            model = self.models_NN[genre]
            # data_loader = self._create_data_loader(tracks)
            # for batch in data_loader:
            out = model(torch.tensor(tracks))
            rating = pd.Series(out.tolist()).apply(lambda x: x[0])
        elif model == 'GB':
            model = self.models_GB[genre]
            rating = list(model.predict(tracks))
        else:
            return 'Wrong model selected'

        tracks_to_playlist = self._tracks.filter(items=['id', 'name', 'id_artist'])
        tracks_to_playlist['coherence'] = rating
        # tracks_to_playlist['coherence'] = tracks_to_playlist['coherence'].apply(lambda x: x[0])
        tracks_to_playlist = tracks_to_playlist.sort_values('coherence', ascending=False)
        return tracks_to_playlist[:length]

    def _filter_genres(self, genre_list, regex_pattern):
        return [genre for genre in genre_list if pd.Series([genre]).str.contains(regex_pattern, regex=True).any()]

    def _get_tracks(self, tracks_path, artists_path, genre) -> pd.DataFrame:
        artists = pd.read_json(artists_path, lines=True)
        filtered_artists = pd.DataFrame()
        for genre in self._genres_to_filter:
            regex_pattern = fr'\b({genre})\b'

            artists['filtered_genres'] = artists['genres'].apply(lambda x: self._filter_genres(x, regex_pattern))
            one_genre = artists[artists['filtered_genres'].apply(lambda x: x != [])]
            one_genre.loc[:, 'genres'] = genre
            filtered_artists = pd.concat([filtered_artists, one_genre], ignore_index=True)

        filtered_artists = filtered_artists.drop(columns=['filtered_genres'])

        tracks = pd.read_json(tracks_path, lines=True)

        genre_frame = filtered_artists[filtered_artists['genres'] == genre]
        genre_artist_ids = genre_frame['id'].tolist()
        tracks =  tracks[tracks['id_artist'].apply(lambda x: x in genre_artist_ids)]
        return tracks



