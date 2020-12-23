from typing import List
from data_models import Model

_MODELS: List[Model] = [
    Model(
        id="COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
        prefix="COCO-Detection",
        name="Faster R-CNN",
        ap=35.7,
        dataset="COCO",
        inferenceTime=0.102,
        details="ResNet 50",
    ),
    Model(
        id="COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        prefix="COCO-Detection",
        name="RetinaNet",
        ap=38.7,
        dataset="COCO",
        inferenceTime=0.041,
        details="ResNet 101 + FPN",
    ),
    Model(
        id="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
        name="Cascade Mask R-CNN",
        ap=44.3,
        dataset="Cityscapes",
        inferenceTime=0.053,
        details="ResNet 50 + FPN",
    ),
]


def get_models() -> List[Model]:
    return _MODELS
