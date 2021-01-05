# Object Detection Benchmarking Tool Backend

## Setup

**Prerequisites**

- CUDA drivers and CUDA-enabled GPU

```
conda env create -f environment.yml
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```

## Running the server

```
uvicorn main:app
```

## API Docs

- Available at http://127.0.0.1:8000/docs