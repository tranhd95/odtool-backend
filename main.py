from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from datasets import get_datasets
from models import get_models
import detectron2


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/datasets")
def send_datasets():
    return get_datasets("datasets")


@app.get("/models")
def send_models():
    return get_models()

