from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from datasets import get_datasets
from models import get_models
from data_models import Dataset, Config


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/datasets")
async def send_datasets():
    return get_datasets("datasets")


@app.get("/models")
async def send_models():
    return get_models()


@app.post("/dataset")
async def set_dataset(dataset: Dataset):
    # TODO set dataset to the benchmark tool
    return {"Response": "Dataset sent successfully."}


@app.post("/configs")
async def receive_configs(configs: List[Config]):
    # TODO set configs to the benchmark tool
    return {"Response": "Configs sent successfully."}


@app.post("/train")
async def start_training():
    pass


@app.get("/console")
async def send_console_output():
    pass
