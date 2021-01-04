import os
import json
import time
from typing import List, Tuple, Dict

import torch
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.engine import HookBase, DefaultTrainer, hooks, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils import comm
from fvcore.nn.precise_bn import get_bn_modules
from datamodels import Dataset, Config, DatasetDict, TrainingState
from datasets import get_categories
from detectron2.structures import BoxMode
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.config import get_cfg
from detectron2 import model_zoo
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from math import isnan
from numpy import argmin
from fastapi.responses import FileResponse


class Benchmark:
    """
    Handles the main logic of the app:
    - Selection of dataset, models, and its parameters.
    - Initiate the training process.
    - Evaluates the trained models.
    - Exports the trained weights.
    """

    def __init__(self):
        self.initialize_state()

    def initialize_state(self):
        self.training_status = TrainingState.beforeTraining
        self.dataset = None
        self.configs = None
        self.split_names = dict()
        self.num_of_categories = -1
        self.test_dicts = None
        self.built_configs = []
        self.imgs = []
        self.img_extension = None

    @property
    def is_ready(self):
        """
        Returns: is ready for training initialization.
        """
        return len(self.configs) > 0 and self.dataset

    def set_dataset(self, dataset: Dataset):
        """
        Register the given dataset. If split ratio is defined, split the datasets by the ratio and register the dataset.
        Args:
            dataset: to be registered
        """
        self.dataset = dataset
        if dataset.splitRatio:
            train, test, val = self._split_dataset(dataset)
            self._register_dataset(train, test, val)
            self.test_dicts = test
        else:
            self._register_dataset()
            self._load_test_dicts()

    @staticmethod
    def _split_dataset(
            dataset: Dataset
    ) -> Tuple[List[DatasetDict], List[DatasetDict], List[DatasetDict]]:
        """
        Splits the dataset by given ratio
        Args:
            dataset: to be split
        Returns: training, testing and validation splits
        """
        all_dict = load_dicts(dataset.name, "all")
        random.shuffle(all_dict)
        ratio = dataset.splitRatio
        split_idx1 = int(dataset.size / 100 * ratio[0])
        split_idx2 = int(dataset.size / 100 * (ratio[0] + ratio[1]))
        train = all_dict[:split_idx1]
        test = all_dict[split_idx1:split_idx2]
        val = all_dict[split_idx2:]
        return train, test, val

    def _register_dataset(self, train=None, test=None, val=None):
        """
        Registers each split to Detectron2's dataset catalog
        Args:
            train: split
            test: split
            val: split
        """
        DatasetCatalog.clear()
        MetadataCatalog.clear()
        category_tuples = get_categories(os.path.join("datasets", self.dataset.name))
        category_names = [tup[1] for tup in category_tuples]
        self.num_of_categories = len(category_names)
        if train and test and val:
            for list_dict, split in zip(
                    [train, test, val], ["train", "test", "validation"]
            ):
                split_name = f"{self.dataset.name}_{split}"
                self.split_names[split] = split_name
                DatasetCatalog.register(split_name, lambda: list_dict)
                MetadataCatalog.get(split_name).set(thing_classes=category_names)
        else:
            for split in ["train", "test", "validation"]:
                split_name = f"{self.dataset.name}_{split}"
                self.split_names[split] = split_name
                DatasetCatalog.register(
                    split_name, lambda: load_dicts(self.dataset.name, split)
                )
                MetadataCatalog.get(split_name).set(thing_classes=category_names)

    def _load_test_dicts(self):
        """
        Loads test dicts to the object for later inference.
        """
        self.test_dicts = load_dicts(self.dataset.name, "test")

    def set_configs(self, configs: List[Config]):
        """
        Sets model configs to the object.
        """
        self.configs = configs

    def start_training(self):
        """
        Initiates training process.
        """
        self.training_status = TrainingState.training
        for config in self.configs:
            self._train(config)
        self.training_status = TrainingState.evaluating
        for built_config in self.built_configs:
            if built_config.EVALUATE_BEST_WEIGHTS:
                built_config.MODEL.WEIGHTS = self._find_best_weights(
                    built_config.OUTPUT_DIR
                )
            else:
                built_config.MODEL.WEIGHTS = os.path.join(
                    built_config.OUTPUT_DIR,
                    "model_final.pth",
                )

    def _train(self, model_config: Config):
        """
        Trains a model defined by the argument.
        Args:
            model_config: to be trained
        """
        cfg = self._build_model_config(model_config)
        self.built_configs.append(cfg)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def _build_model_config(self, model_config: Config):
        """
        Builds Detectron2's model config according to given model config
        Args:
            model_config:
        Returns: the built Detectron2's config
        """
        params = model_config.parameters
        cfg = get_cfg()
        cfg.EVALUATE_BEST_WEIGHTS = model_config.parameters.saveBestWeights
        cfg.MODEL_NAME = model_config.name
        cfg.merge_from_file(model_zoo.get_config_file(model_config.id))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config.id)
        timestamp = time.strftime("%y%m%d_%H_%M_%S")
        model_output_dir = f"{timestamp}_{model_config.name.replace(' ', '_')}"
        cfg.OUTPUT_DIR = os.path.join("outputs", model_output_dir)
        cfg.DATASETS.TRAIN = (self.split_names["train"],)
        cfg.DATASETS.TEST = (self.split_names["test"],)
        cfg.DATASETS.VAL = (self.split_names["validation"],)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_of_categories
        cfg.MODEL.RETINANET.NUM_CLASSES = self.num_of_categories
        cfg.MODEL.MASK_ON = False
        cfg.SOLVER.IMS_PER_BATCH = params.batchSize
        one_epoch = int(self.dataset.trainSize / params.batchSize)
        cfg.SOLVER.MAX_ITER = params.epochs * one_epoch
        cfg.SOLVER.BASE_LR = params.learningRate
        cfg.SOLVER.CHECKPOINT_PERIOD = params.checkpointPeriod
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        return cfg

    @staticmethod
    def _find_best_weights(output_dir: str):
        """
        Finds the best weights w.r.t. values of validation loss.
        Args:
            output_dir: to be scanned
        Returns: the path to the best weights
        """
        metrics = load_metrics_json(os.path.join(output_dir, "metrics.json"))
        val_losses = [
            line["total_val_loss"] for line in metrics if "total_val_loss" in line
        ]
        index = argmin(val_losses)
        if index == len(val_losses) - 1:
            best_model = "model_final.pth"
        else:
            best_model = f"model_{str(metrics[index]['iteration']).zfill(7)}.pth"
        return os.path.join(output_dir, best_model)

    def random_predict(self, threshold: float):
        """
        Randomly selects a test image and run inference on it with every trained model.
        Args:
            threshold: If prediction's score is higher than threshold, it is true positive
        Returns: the chosen image's path
        """
        self.imgs = []
        random_dict: DatasetDict = random.choice(self.test_dicts)
        img_path = random_dict["file_name"]
        root, self.img_extension = os.path.splitext(img_path)
        random_cv2img = cv2.imread(img_path)
        metadata = MetadataCatalog.get(self.dataset.name + "_test")
        visualizer_gt = Visualizer(
            random_cv2img[:, :, ::-1], metadata=metadata, scale=1
        )
        vis_gt = visualizer_gt.draw_dataset_dict(random_dict)
        gt_img = vis_gt.get_image()[:, :, ::-1]
        self.imgs.append(gt_img)
        for config in self.built_configs:
            config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
            config.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
            predictor = DefaultPredictor(config)
            visualizer_pred = Visualizer(
                random_cv2img[:, :, ::-1], metadata=metadata, scale=1
            )
            predictions = predictor(random_cv2img)
            vis_pred = visualizer_pred.draw_instance_predictions(
                predictions["instances"].to("cpu")
            )
            self.imgs.append(vis_pred.get_image()[:, :, ::-1])
        return random_dict["file_name"]

    def evaluate(self):
        """
        Initiates evaluation process of trained models.
        Returns: the evaluation results.
        """
        evaluation = dict({"headers": [], "items": []})
        for config in self.built_configs:
            config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
            config.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
            predictor = DefaultPredictor(config)
            coco_path = os.path.join(
                config.OUTPUT_DIR, f"{self.dataset.name}_coco_format.json"
            )
            test_split = f"{self.dataset.name}_test"
            convert_to_coco_json(test_split, coco_path, False)
            evaluator = COCOEvaluator(
                test_split, config, False, output_dir=config.OUTPUT_DIR
            )
            test_loader = build_detection_test_loader(config, test_split)
            metrics_dict = inference_on_dataset(
                predictor.model, test_loader, evaluator
            )["bbox"]
            for m in metrics_dict:
                if isnan(metrics_dict[m]):
                    metrics_dict[m] = -1
                else:
                    metrics_dict[m] = round(metrics_dict[m], 2)
            metrics_dict["Model"] = config.MODEL_NAME
            evaluation["items"].append(metrics_dict)
        evaluation["headers"].append(
            {"text": "Model", "value": "Model", "sortable": False, "align": "start"}
        )
        for m in evaluation["items"][0]:
            if m != "Model":
                evaluation["headers"].append({"text": m, "value": m})
        return evaluation

    def send_weights(self, model_id):
        cfg = self.built_configs[int(model_id)]
        return FileResponse(cfg.MODEL.WEIGHTS, filename=cfg.MODEL_NAME.replace(" ", "_") + ".pth",
                            media_type="application/pth")


def load_dicts(dataset_name: str, json_name: str) -> List[DatasetDict]:
    """
    Loads a Detectron2 Stanard Dataset dicts from a json file.
    Args:
        json_name: Path to a .json file with a list of dicts

    Returns:
        List[DatasetDict] of images with annotations (Detectron2 Standard Dataset dicts)
    """
    path_to_json = os.path.join("datasets", dataset_name, json_name + ".json")
    with open(path_to_json, "r") as file:
        list_of_dicts = json.load(file)
    for img_dict in list_of_dicts:
        img_dict["file_name"] = os.path.join(
            "datasets", dataset_name, "images", img_dict["file_name"]
        )
        for annotation in img_dict["annotations"]:
            bbox_mode_num = annotation["bbox_mode"]
            if bbox_mode_num == 0:
                annotation["bbox_mode"] = BoxMode.XYXY_ABS
            elif bbox_mode_num == 1:
                annotation["bbox_mode"] = BoxMode.XYWH_ABS
            else:
                raise Exception(
                    "Bbox modes other than BoxMode.XYXY_ABS (0) and BoxMode.XYWH_ABS (1) are not yet "
                    "supported."
                )
    return list_of_dicts


class ValidationLoss(HookBase):
    """
    A Detectron2 hook that displays validation loss
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                "val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced, **loss_dict_reduced
                )


class Trainer(DefaultTrainer):
    """
    Detectron2's default trainer with correct periodical checkpointing.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_hooks([ValidationLoss(cfg)])
        self._hooks = self._hooks[:-2] + self._hooks[-2:][::-1]

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(
                hooks.PeriodicWriter(
                    self.build_writers(), period=cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )  # <- Overwriten here
        return ret


def load_metrics_json(json_path) -> List[Dict]:
    """
    Loads metrics.json to list of dictionaries
    Args:
        json_path: to metrics.json
    Returns: list of dictionaries
    """
    lines = []
    with open(json_path, "r") as f:
        for line in f:
            lines.append(json.loads(line))
    return lines
