import os
import glob
import json
from typing import List, Tuple
from datamodels import Dataset


def get_datasets(directory: str = "datasets") -> List[Dataset]:
    """
    Scans the datasets folder for any available datasets.
    Args:
        directory: Path to datasets directory
    Returns: List of datasets.
    """
    datasets: List[Dataset] = []
    dataset_directories = [p.path for p in os.scandir(directory) if p.is_dir()]
    for dataset_dir in dataset_directories:
        errors = []
        is_valid = True
        dataset_contents = os.listdir(dataset_dir)
        has_all = "all.json" in dataset_contents
        has_splits = {"test.json", "train.json", "validation.json"}.issubset(
            set(dataset_contents)
        )
        train_size = get_train_size(dataset_dir) if has_splits else 0

        try:
            categories = get_categories(dataset_dir)
        except FileNotFoundError:
            errors.append(
                "Names of categories not found. Please provide category_names.csv."
            )
            categories = []
            is_valid = False

        images_count = get_number_of_images(dataset_dir)
        categories_count = len(categories)

        if not has_all and not has_splits:
            errors.append(
                "No annotations found. Please provide all.json or train.json, test.json and validation.json"
            )
            is_valid = False

        if categories_count < 1:
            errors.append("There are no categories in category_names.csv.")
            is_valid = False

        if images_count < 1:
            errors.append("There are no images in the images folder.")
            is_valid = False

        dataset: Dataset = Dataset(
            name=os.path.basename(dataset_dir),
            size=images_count,
            categories=categories_count,
            hasAll=has_all,
            hasSplits=has_splits,
            isValid=is_valid,
            errors=errors,
            trainSize=train_size,
        )
        datasets.append(dataset)
    return datasets


def get_categories(dataset_name: str) -> List[Tuple[int, str]]:
    """
    Args:
        dataset_name: dataset's name
    Returns: List of category tuples (id, category_name) of the given dataset.
    """
    rows = []
    with open(os.path.join(dataset_name, "category_names.csv"), "r") as csv:
        for row in csv.readlines():
            values = row.split(",")
            tpl = (int(values[0]), values[1].strip())
            rows.append(tpl)
    return sorted(rows, key=lambda tup: tup[0])


def get_number_of_images(dataset_path: str) -> str:
    """
    Args:
        dataset_path: path to a given dataset
    Returns: the total number of images of a given dataset.
    """
    count = 0
    for path in glob.iglob(f"{dataset_path}/images/**", recursive=True):
        if os.path.isfile(path):
            count += 1
    return count


def get_train_size(dataset_path: str) -> int:
    """
    Args:
        dataset_path: to a given dataset
    Returns: the training size of a given dataset specified in train.json
    """
    with open(os.path.join(dataset_path, "train.json"), "r") as f:
        dct = json.load(f)
    return len(dct)
