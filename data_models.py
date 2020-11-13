from typing import List
from pydantic import BaseModel


class Dataset(BaseModel):
    name: str
    size: int
    categories: int
    hasAll: bool
    hasSplits: bool
    isValid: bool
    errors: List[str]
    trainSize: int


class Model(BaseModel):
    id: str
    name: str
    ap: float
    dataset: str
    inferenceTime: float
    details: str
