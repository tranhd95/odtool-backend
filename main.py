from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, BackgroundTasks, HTTPException
from starlette.responses import StreamingResponse
from datasets import get_datasets
from models import get_models
from datamodels import Dataset, Config, PredictParams
import sys
from benchmark import Benchmark
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
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
    """
    Scans the datasets folder and returns a list of datasets. Invalid datasets will be returned with warning.
    Returns: List of available datasets.
    """
    return get_datasets("datasets")


@app.get("/models")
async def send_models():
    """
    Returns: Available object detection models and its stats.
    """
    return get_models()


@app.post("/dataset")
async def set_dataset(dataset: Dataset):
    """
    Sets a dataset for benchmarking.
    Args:
        dataset: Dataset to be set.
    Returns: Successful response if it went ok.
    """
    benchmark.set_dataset(dataset)
    return {"Response": "Dataset sent successfully."}


@app.post("/configs")
async def set_configs(configs: List[Config]):
    """
    Sets the model configs.
    Args:
        configs: Chosen models with selected parameters.
    Returns: Successful response if it went ok.
    """
    benchmark.set_configs(configs)
    return {"Response": "Configs sent successfully."}


@app.post("/train")
async def heavy_task(bg_tasks: BackgroundTasks):
    """
    Initiate the training process
    Args:
        bg_tasks: BackgroundTasks class for setting background tasks (see FastAPI docs)
    Returns: Successful response it it went ok.
    """
    if benchmark.is_ready:
        sys.stdout = out
        bg_tasks.add_task(start_training)
        return {"Response": "Training..."}
    else:
        raise HTTPException(
            status_code=405,
            detail="Cannot start training without selected dataset and models.",
        )


def start_training():
    """
    Initiates training
    """
    benchmark.start_training()


@app.get("/console")
async def send_console_output():
    """
    Returns: The standard output.
    """
    return {"stdout": out.getvalue(), "trainingStatus": benchmark.training_status}


@app.get("/evaluate")
async def evaluate():
    """
    Initiate evaluation of the trained models.
    Returns: Evaluation results.
    """
    sys.stdout = sys.__stdout__
    json_compatible_item_data = jsonable_encoder(benchmark.evaluate())
    return JSONResponse(content=json_compatible_item_data)


@app.post("/predict")
async def random_predict(params: PredictParams):
    """
    Randomly pick a test image and run inference on every trained models and save the images to benchmark.imgs.
    Args:
        params: Score threshold.
    Returns: Successful response if it went ok.
    """
    return {"Response": benchmark.random_predict(params.threshold)}


@app.get("/image")
async def send_image(index: int):
    """
    Send the inferred images.
    Args:
        index: of trained model.
    Returns: Predicted image
    """
    img = benchmark.imgs[index]
    res, im_jpg = cv2.imencode(benchmark.img_extension, img)
    return StreamingResponse(
        io.BytesIO(im_jpg.tobytes()), media_type=f"image/{benchmark.img_extension}"
    )


@app.get("/weights/{index}")
async def send_weights(index):
    """
    Sends the index-th trained model's weights
    Args:
        index: the order of the model
    Returns: .pth file with weights
    """
    return benchmark.send_weights(index)
