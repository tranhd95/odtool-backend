from typing import List
from data_models import Model

_MODELS: List[Model] = [
    Model(
        id="faster_rcnn_R_101_FPN_3x",
        name="Faster R-CNN",
        ap=42.0,
        dataset="COCO",
        inferenceTime=0.051,
        details="ResNet 101 + FPN",
    ),
    Model(
        id="retinanet_R_101_FPN_3x",
        name="RetinaNet",
        ap=40.4,
        dataset="COCO",
        inferenceTime=0.054,
        details="ResNet 101 + FPN",
    ),
    Model(
        id="cascade_mask_rcnn_R_50_FPN_3x",
        name="Cascade Mask R-CNN",
        ap=44.3,
        dataset="Cityscapes",
        inferenceTime=0.053,
        details="ResNet 50 + FPN",
    ),
]


def get_models() -> List[Model]:
    return _MODELS
