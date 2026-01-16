from torch.utils.data import Dataset

from sign_ml.data import TrafficSignsDataset


def test_traffic_signs_dataset_is_dataset() -> None:
    """TrafficSignsDataset should implement the PyTorch Dataset interface."""

    dataset = TrafficSignsDataset("train")
    assert isinstance(dataset, Dataset)
