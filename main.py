import torchvision.models as models
from src import VERSION
from src.export import Export

if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    net.cpu()
    net.eval()

    print(f'Repo Version: {VERSION}.')

    args = {
        'model': net,
        'mode': 'onnx',
        'shape': (1, 3, 224, 224),
        'opset_version': 13,
        'output': './',
        'input_names': ['images'],
        'output_names': ['output'],
        'dynamic_axes': None,
        'is_simplify': True,
    }

    export = Export(**args)
    export.run()
