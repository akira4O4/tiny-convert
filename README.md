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

pip install onnx onnxsim
python main.py
```

---

## PyTorch -> ONNX

If your need dynamic shape

```python
dynamic_shape = {
    'images': {0: 'batch'},
    'output': {0: 'batch'}
}
```

If your model has multi input or output

```python
input_names = ['images1', 'images']
output_names = ['output1', 'output2']
```

Run

```python
if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    net.cpu()
    net.eval()

    # if you need dynamic shape
    dynamic_shape = {
        'images': {0: 'batch'},
        'output': {0: 'batch'},
    }

    args = {
        'model': net,
        'mode': 'onnx',
        'shape': (1, 3, 224, 224),  # NCHW
        'opset_version': 13,
        'output': './',
        'input_names': ['images'],
        'output_names': ['output'],
        'dynamic_axes': dynamic_shape,
        'is_simplify': True,
    }

    export = Export(**args)
    export.run()

```

---

## PyTorch -> TorchScript

Run

```python
if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    net.cpu()
    net.eval()

    args = {
        'model': net,
        'mode': 'torchscript',
        'shape': (1, 3, 224, 224),  # NCHW
        'output': './',
    }

    export = Export(**args)
    export.run()
```
