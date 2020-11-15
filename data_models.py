from typing import List, Optional, Union
from pydantic import BaseModel
from enum import Enum


class Dataset(BaseModel):
    name: str
    size: int
    categories: int
    hasAll: bool
    hasSplits: bool
    isValid: bool
    errors: List[str]
    trainSize: int
    splitRatio: Optional[List[int]]


class Model(BaseModel):
    id: str
    name: str
    ap: float
    dataset: str
    inferenceTime: float
    details: str


class Parameters(BaseModel):
    epochs: int
    batchSize: int
    checkpointPeriod: int
    learningRate: float
    saveBestWeights: bool


class Config(BaseModel):
    id: str
    name: str
    parameters: Parameters


class Annotation(BaseModel):
    bbox: List[int]
    bbox_mode: int
    category_id: int


class DatasetDict(BaseModel):
    file_name: str
    height: int
    width: int
    image_id: Union[int, str]
    annotations: List[Annotation]


class TrainingState(Enum):
    beforeTraining = 1
    training = 2
    evaluating = 3
    done = 4


class PredictParams(BaseModel):
    threshold: float
