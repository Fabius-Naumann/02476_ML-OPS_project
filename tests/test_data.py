import inspect
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

import sign_ml.data as data_module
from sign_ml import PROCESSED_DIR


def _find_dataset_class():
    """Find the Dataset subclass defined in sign_ml.data."""
    for _, obj in inspect.getmembers(data_module, inspect.isclass):
        if issubclass(obj, Dataset) and obj is not Dataset:
            return obj
    return None


@pytest.mark.skipif(not Path.exists(PROCESSED_DIR), reason="Processed training data not available yet")
def test_dataset_basic_functionality():
    DatasetClass = _find_dataset_class()
    assert DatasetClass is not None, "No Dataset subclass found in sign_ml.data"

    dataset = data_module.TrafficSignsDataset("train")

    assert isinstance(dataset, Dataset), "Dataset instance is not of type torch.utils.data.Dataset"
    assert len(dataset) > 0, "Dataset length should be greater than 0"

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor), "Data sample x should be a torch.Tensor"
    assert isinstance(y, torch.Tensor), "Data sample y should be a torch.Tensor"
