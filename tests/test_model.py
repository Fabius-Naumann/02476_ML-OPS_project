import inspect
import torch
import torch.nn as nn

import sign_ml.model as model_module


def _find_model_class():
    """Find the nn.Module subclass defined in sign_ml.model."""
    for _, obj in inspect.getmembers(model_module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            return obj
    return None


def _instantiate_model(ModelClass):
    """Instantiate model, providing num_classes if required."""
    sig = inspect.signature(ModelClass.__init__)
    params = sig.parameters

    if "num_classes" in params:
        return ModelClass(num_classes=3)
    return ModelClass()


def test_model_instantiation_and_forward():
    ModelClass = _find_model_class()
    assert ModelClass is not None, "No nn.Module subclass found in sign_ml.model"

    model = _instantiate_model(ModelClass)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    assert isinstance(y, torch.Tensor)
