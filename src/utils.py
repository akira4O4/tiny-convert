import os
import torch


def load_weight(model, path: str) -> None:
    assert model is not None, 'Model is None.'
    assert os.path.exists(path) is True, 'Weight path is not found.'

    checkpoint = torch.load(path)
    static_dict = checkpoint['state_dict']

    model.load_state_dict(static_dict, strict=False)
    model.cpu()
    model.eval()
