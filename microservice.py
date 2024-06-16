from typing import Annotated
from Playlist import Playlist
from RegresionModel import RegresionModel
from fastapi import FastAPI, HTTPException, Query, Path

app = FastAPI()


# setting up the microservice: python -m uvicorn microservice:app --reload
# example of using the microservice: curl http://127.0.0.1:8000/generate/genre=pop&num_tracks=50&model=NN


@app.get("/status")
async def check_status():
    """
    Allows for checking server status

    :return: ok status if server is running
    """
    return {"status": "OK"}

# two models, GB and NN
@app.get("/generate", response_model=list)
async def generate_neural(genre: str, num_tracks: int, model: str):
    models_NN = {}
    models_GB = {}
    genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']
    for genre in genres_to_filter:
        models_NN[genre] = f'state_dicts/{genre}_model_dict.pt'
    for genre in genres_to_filter:
        models_GB[genre] = f'state_dicts_GB/{genre}_state_dict.joblib'
    playlist = Playlist(models_NN, models_GB)
    return playlist.create_playlist(genre, num_tracks, 'tracks.jsonl', 'artists.jsonl', model).to_numpy().tolist()


@app.post("/generate/ab")
async def ab_experiment():
    pass
