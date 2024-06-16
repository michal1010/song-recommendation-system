from typing import Annotated
from Playlist import Playlist
from RegresionModel import RegresionModel
from fastapi import FastAPI, HTTPException, Query, Path
from configuration import NN_DICTS_PATH, GB_DICTS_PATH
import datetime


app = FastAPI()


# setting up the microservice: python -m uvicorn microservice:app --reload
# example of using the microservice: curl http://127.0.0.1:8000/generate/genre=pop&num_tracks=50&model=NN


@app.get("/status")
async def check_status():
    """
    Allows for checking server status

    :return: ok status if server is running
    """
    with open('microservice_log.txt', 'a') as log:
        log.write(f"Microservice status has been checked ({datetime.datetime.now()})\n")

    return {"status": "OK"}

@app.get("/generate", response_model=list)
async def generate_neural(genre: str, num_tracks: int, model: str):
    with open('microservice_log.txt', 'a') as log:
        log.write(f"Generate {genre} playlist with {num_tracks} using {model} model ({datetime.datetime.now()})\n")
    models_NN = {}
    models_GB = {}
    genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']
    for genre in genres_to_filter:
        models_NN[genre] = f'{NN_DICTS_PATH}/{genre}_model_dict.pt'
    for genre in genres_to_filter:
        models_GB[genre] = f'{GB_DICTS_PATH}/{genre}_state_dict.joblib'
    playlist = Playlist(models_NN, models_GB)
    result = playlist.create_playlist(genre, num_tracks, 'tracks.jsonl', 'artists.jsonl', model).to_numpy().tolist()
    print(result)
    with open('microservice_log.txt', 'a') as log:
        log.write(f"Result:\n")
        for element in result:
            for value in element:
                try:
                    log.write(f"{value}, ")
                except:
                    pass
            log.write(f"\n")
        log.write(f"\n")
    return result


@app.get("/abtest", response_model=dict)
async def ab_experiment(genre: str, num_tracks: int):
    with open('microservice_log.txt', 'a') as log:
        log.write(f"Run AB Test, generating {genre} playlist with {num_tracks} ({datetime.datetime.now()})\n")
    models_NN = {}
    models_GB = {}
    genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']
    for genre in genres_to_filter:
        models_NN[genre] = f'{NN_DICTS_PATH}/{genre}_model_dict.pt'
    for genre in genres_to_filter:
        models_GB[genre] = f'{GB_DICTS_PATH}/{genre}_state_dict.joblib'
    playlist = Playlist(models_NN, models_GB)

    NN_playlist = playlist.create_playlist(genre, num_tracks, 'tracks.jsonl', 'artists.jsonl', model='NN').to_numpy().tolist()
    GB_playlist = playlist.create_playlist(genre, num_tracks, 'tracks.jsonl', 'artists.jsonl', model='GB').to_numpy().tolist()


    result_dict = {
        "Neural Networks"   : NN_playlist,
        "Gradient Boost"    : GB_playlist
        }
    with open('microservice_log.txt', 'a') as log:
        log.write(f"Playlist generated using Neural Networks model:\n")
        for element in NN_playlist:
            for value in element:
                try:
                    log.write(f"{value}, ")
                except:
                    pass
            log.write(f"\n")
        log.write(f"\n")
        log.write(f"Playlist generated using Gradient Boost model:\n")
        for element in GB_playlist:
            for value in element:
                try:
                    log.write(f"{value}, ")
                except:
                    pass
            log.write(f"\n")
        log.write(f"\n")

    return result_dict
