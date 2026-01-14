"""Local corrupt-MNIST training package for the experiment-logging exercises."""

from .data import corrupt_mnist
from .model import MyAwesomeModel

__all__ = ["corrupt_mnist", "MyAwesomeModel"]
