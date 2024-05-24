from typing import Annotated
from Playlist import Playlist
from RegresionModel import RegresionModel
from fastapi import FastAPI, HTTPException, Query, Path

app = FastAPI()


# setting up the microservice: python -m uvicorn microservice:app --reload
# example of using the microservice: curl http://127.0.0.1:8000/status


@app.get("/status")
async def check_status():
    """
    Allows for checking server status

    :return: ok status if server is running
    """
    return {"status": "OK"}


@app.get("/generate/neural", response_model=list)
async def generate_neural(genre: str, num_tracks: int):
    models = {}
    genres_to_filter = ['pop', 'trap', 'rap', 'rock', 'metal', 'hip hop']
    for genre in genres_to_filter:
        models[genre] = f'state_dicts/{genre}_model_dict.pt'
    playlist = Playlist(models, RegresionModel)
    return playlist.create_playlist(genre, num_tracks, 'tracks.jsonl', 'artists.jsonl').to_numpy().tolist()


@app.get("/generate/sessions")
async def generate_sessions():
    pass


@app.post("/generate/ab")
async def ab_experiment():
    pass
