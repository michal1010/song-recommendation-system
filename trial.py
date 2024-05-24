from Playlist import Playlist


from RegresionModel import RegresionModel


models = {}
genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']



for genre in genres_to_filter:
    models[genre] = 'pop_model_dict.pt'
playlist = Playlist(models, RegresionModel, 'tracks.jsonl')
print(playlist.create_playlist('rock', 15, 'tracks.jsonl', 'artists.jsonl'))