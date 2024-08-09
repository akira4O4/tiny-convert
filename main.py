import torchvision.models as models

from src import VERSION
from src.export import Export
from src.utils import load_weight

if __name__ == '__main__':
    # Prepare your model -----------------------------------------------------------------------------------------------
    model = models.resnet18(pretrained=True)

    print(f'Repo Version: {VERSION}.')

    # Convert model ----------------------------------------------------------------------------------------------------
    args = {
        'model': model,
        'mode': 'onnx',
        'shape': (1, 3, 256, 256),  # NCHW
        'opset_version': 12,
        'output': './',
        'input_names': ['images'],
        'output_names': ['output0'],
        'dynamic_axes': None,
        'is_simplify': True,
    }

    export = Export(**args)
    export.run()
