# Tiny Model Convert Tool

---

## Introduction

This Repo is a tiny code for convert pytorch mode to other types model.

If have any questions or suggestions,please email me.

---

## Feat

- Support Windows,Linux,MacOS
- Convert **PyTorch** model to **ONNX** model (**Support dynamic shape**)
- Convert **PyTorch** model to **TorchScript** model

---

## How to install

```bash
cd <your work dir>
git clone https://github.com/akira4O4/tiny-convert.git
cd tiny-convert

pip install -r requirements.txt
```

---

## Load weight

```python
from src.utils import load_weight

model = model(...)
model_path = r''
load_weight(model, model_path)
```

---

## PyTorch → ONNX

If you need `dynamic shape`

```python
dynamic_axes = {
    'images': {0: 'batch'},
    'output': {0: 'batch'},
    ...
}
```

If you need `multi` inputs and outputs

```python
input_names = ['images1', 'images2', ...]
output_names = ['output1', 'output2', ...]
```

Run

```python
import torchvision.models as models

from src import VERSION
from src.export import Export
from src.utils import load_weight

if __name__ == '__main__':
    # Create your model
    net = models.resnet18(pretrained=True)

    args = {
        'model': net,
        'mode': 'onnx',
        'shape': (1, 3, 224, 224),  # NCHW
        'opset_version': 13,
        'output': './',
        'input_names': ['images'],
        'output_names': ['output'],
        'dynamic_axes': None,
        'is_simplify': True,
    }

    export = Export(**args)
    export.run()

```

---

## PyTorch → TorchScript

Run

```python
import torchvision.models as models

from src import VERSION
from src.export import Export
from src.utils import load_weight

if __name__ == '__main__':
    net = models.resnet18(pretrained=True)

    args = {
        'model': net,
        'mode': 'torchscript',
        'shape': (1, 3, 224, 224),  # NCHW
        'output': './',
    }

    export = Export(**args)
    export.run()
```
