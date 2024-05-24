from typing import Annotated

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


@app.get("/generate/neural")
async def generate_neural():
    pass


@app.get("/generate/sessions")
async def generate_sessions():
    pass


@app.post("/generate/ab")
async def ab_experiment():
    pass
