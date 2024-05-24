from Playlist import Playlist


from RegresionModel import RegresionModel


models = {}
genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']



for genre in genres_to_filter:
    models[genre] = f'state_dicts/{genre}_model_dict.pt'

playlist = Playlist(models, RegresionModel)
print(playlist.create_playlist('pop', 15, 'tracks.jsonl', 'artists.jsonl'))