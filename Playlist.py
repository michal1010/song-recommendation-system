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

class Playlist:

    def __init__(self, models_parameters: dict, model_name, tracks_path) -> None:
        self.models = self._create_models(models_parameters, model_name)
        self._genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']

    def _create_models(self, models_parameters: dict, model_name):
        models = {}
        for genre, parameters_path in models_parameters.items():
            model = model_name()
            model.load_state_dict(torch.load(parameters_path))
            model.eval()
            models[genre] = model

    def create_playlist(self, genre, length, tracks_path):
        if genre in self.models.keys():
            model = self.models[genre]
        else:
            return 'Wrong genre selected choose the one from : pop, trap' \
                    + 'rap, rock, metal, hip hop'
        self._tracks = self._get_tracks(tracks_path, genre)

        tracks = self._tracks.drop(["id", 'name', 'explicit', 'id_artist', 'key', 'release_date'], axis=1)
        rating = model(tracks)
        tracks_to_playlist = self._tracks.filter(items=['id', 'name', 'id_artist'])
        tracks_to_playlist['coherence'] = rating
        tracks_to_playlist.sort_values()
        return tracks_to_playlist.loc[:, :length]

    def _filter_genres(genre_list, regex_pattern):
        return [genre for genre in genre_list if pd.Series([genre]).str.contains(regex_pattern, regex=True).any()]

    def _get_tracks(self, tracks_path, genre) -> pd.DataFrame:
        artists = pd.read_json('artists.jsonl', lines=True)
        filtered_artists = pd.DataFrame()
        for genre in self._genres_to_filter:
            regex_pattern = fr'\b({genre})\b'

            artists['filtered_genres'] = artists['genres'].apply(lambda x: self._filter_genres(x, regex_pattern))
            one_genre = artists[artists['filtered_genres'].apply(lambda x: x != [])]
            one_genre.loc[:, 'genres'] = genre
            filtered_artists = pd.concat([filtered_artists, one_genre], ignore_index=True)

        filtered_artists = filtered_artists.drop(columns=['filtered_genres'])

        tracks = pd.read_json('tracks.jsonl', lines=True)

        genre_frame = filtered_artists[filtered_artists['genres'] == genre]
        genre_artist_ids = genre_frame['id'].tolist()
        tracks =  tracks[tracks['id_artist'].apply(lambda x: x in genre_artist_ids)]
        return tracks.drop(["id", 'name', 'explicit', 'id_artist', 'key', 'release_date'], axis=1)



