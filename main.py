from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, BackgroundTasks, HTTPException
from starlette.responses import StreamingResponse
from datasets import get_datasets
from models import get_models
from data_models import Dataset, Config, PredictParams
from benchmark import Benchmark
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import sys
import io
import cv2

benchmark = Benchmark()
out = io.StringIO()
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
    benchmark.set_dataset(dataset)
    return {"Response": "Dataset sent successfully."}


@app.post("/configs")
async def receive_configs(configs: List[Config]):
    benchmark.set_configs(configs)
    return {"Response": "Configs sent successfully."}


@app.post("/train")
async def heavy_task(bg_tasks: BackgroundTasks):
    if benchmark.isReady:
        sys.stdout = out
        bg_tasks.add_task(start_training)
        return {"Response": "Training..."}
    else:
        raise HTTPException(
            status_code=405,
            detail="Cannot start training without selected dataset and models.",
        )


@app.get("/console")
async def send_console_output():
    return {"stdout": out.getvalue(), "trainingStatus": benchmark.training_status}


@app.get("/evaluate")
async def evaluate():
    sys.stdout = sys.__stdout__
    json_compatible_item_data = jsonable_encoder(benchmark.evaluate())
    return JSONResponse(content=json_compatible_item_data)


@app.post("/predict")
async def random_predict(params: PredictParams):
    return {"Response": benchmark.random_predict(params.threshold)}


@app.get("/image")
async def send_image(index: int):
    img = benchmark.imgs[index]
    res, im_jpg = cv2.imencode(benchmark.img_extension, img)
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type=f"image/{benchmark.img_extension}")


def start_training():
    benchmark.start_training()
