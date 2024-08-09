import os
from typing import Tuple, Optional

import onnx
import torch
from onnxsim import simplify


class Export:
    def __init__(
        self,
        model=None,
        mode: Optional[str] = 'onnx',  # noqa
        shape: Tuple[int] = (1, 3, 224, 224),
        opset_version: Optional[int] = 12,
        output: Optional[str] = None,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        dynamic_axes: Optional[dict] = None,
        is_simplify: Optional[bool] = True,

    ):
        self.model = model
        self.mode = mode
        self.shape = shape
        self.opset_version = opset_version

        self.output = output
        self.save_path = output

        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.is_simplify = is_simplify

        if self.input_names is None:
            self.input_names = ['images']

        if self.output_names is None:
            self.output_names = ['images']

        assert self.mode in ['onnx', 'torchscript'], 'mode type error'

        if self.mode == 'onnx':
            if os.path.isdir(self.save_path):
                self.save_path = os.path.join(self.output, 'model.onnx')

        elif self.mode == 'torchscript':
            if os.path.isdir(self.save_path):
                self.save_path = os.path.join(self.output, 'model.torchscript')

    def make_dummy_input(self) -> torch.Tensor:
        assert len(self.shape) == 4, 'shape.len()!=4.'
        x = torch.rand(self.shape)
        return x

    def run(self) -> None:
        print(f'Export Mode: {self.mode}.')
        print(f'Model Input Shape: {self.shape}.')

        if self.mode == 'onnx':
            print('Start Export ONNX Model...')
            self.export_onnx()
            print('Export ONNX Model End.')

        elif self.mode == 'torchscript':
            print('Start Export TorchScript Model...')
            self.export_torchscript()
            print('Export TorchScript Model End.')

        print(f'Model Output: {self.save_path}.')

    def export_torchscript(self) -> None:  # noqa
        self.model.eval()
        dummy_input = self.make_dummy_input()
        model_torchscript = torch.jit.trace(self.model, dummy_input, strict=False)
        model_torchscript.save(self.save_path)

    def export_onnx(self) -> None:
        torch.onnx.export(
            self.model,
            self.make_dummy_input(),
            self.save_path,
            opset_version=self.opset_version,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes
        )

        onnx_model = onnx.load(self.save_path)
        onnx.checker.check_model(onnx_model)

        if self.is_simplify:
            onnx_model, _ = simplify(onnx_model)
            onnx.save(onnx_model, self.save_path)

    def torchscript2onnx(self) -> None:
        ...
