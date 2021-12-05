# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

"""
import logging
import os
import pathlib
import zipfile

from bentoml.exceptions import (
    InvalidArgument,
    MissingDependencyException,
)
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv
from bentoml.utils import cloudpickle

logger = logging.getLogger(__name__)


def _is_path_like(path):
    return isinstance(path, (str, bytes, pathlib.Path, os.PathLike))


def _is_pytorch_lightning_model_file_like(path):
    return (
            _is_path_like(path)
            and os.path.isfile(path)
            and str(path).lower().endswith(".pt")
    )


class PytorchModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving/loading objects with torch.save and torch.load
    Args:
        name (string): name of the artifact
    Raises:
        MissingDependencyException: torch package is required for PytorchModelArtifact
        InvalidArgument: invalid argument type, model being packed must be instance of
            torch.nn.Module
    Example usage:
    >>> import torch.nn as nn
    >>>
    >>> class Net(nn.Module):
    >>>     def __init__(self):
    >>>         super(Net, self).__init__()
    >>>         ...
    >>>
    >>>     def forward(self, x):
    >>>         ...
    >>>
    >>> net = Net()
    >>> # Train model with data
    >>>
    >>>
    >>> import bentoml
    >>> from bentoml.adapters import ImageInput
    >>> from bentoml.tkitBentomlFrameworksExpand.pytorch import PytorchModelArtifact
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([PytorchModelArtifact('net')])
    >>> class PytorchModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=ImageInput(), batch=True)
    >>>     def predict(self, imgs):
    >>>         outputs = self.artifacts.net(imgs)
    >>>         return outputs
    >>>
    >>>
    >>> svc = PytorchModelService()
    >>>
    >>> # Pytorch model can be packed directly.
    >>> svc.pack('net', net)
    >>>
    >>> # Alternatively,
    >>>
    >>> # Pack a TorchScript Model
    >>> # Random input in the format expected by the net
    >>> sample_input = ...
    >>> traced_net = torch.jit.trace(net, sample_input)
    >>> svc.pack('net', traced_net)
    """

    def __init__(self, name, file_extension=".pt"):
        super().__init__(name)
        self._file_extension = file_extension
        self._model = None

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + self._file_extension)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-renamed
        try:
            import torch
        except ImportError:
            raise MissingDependencyException(
                "torch package is required to use PytorchModelArtifact"
            )

        if not isinstance(model, torch.nn.Module):
            raise InvalidArgument(
                "PytorchModelArtifact can only pack type \
                'torch.nn.Module' or 'torch.jit.ScriptModule'"
            )

        self._model = model
        return self

    def load(self, path):
        try:
            import torch
        except ImportError:
            raise MissingDependencyException(
                "torch package is required to use PytorchModelArtifact"
            )

        # TorchScript Models are saved as zip files
        if zipfile.is_zipfile(self._file_path(path)):
            model = torch.jit.load(self._file_path(path))
        else:
            model = cloudpickle.load(open(self._file_path(path), 'rb'))

        if not isinstance(model, torch.nn.Module):
            raise InvalidArgument(
                "Expecting PytorchModelArtifact loaded object type to be "
                "'torch.nn.Module' or 'torch.jit.ScriptModule' \
                but actually it is {}".format(
                    type(model)
                )
            )

        return self.pack(model)

    def set_dependencies(self, env: BentoServiceEnv):
        logger.warning(
            "BentoML by default does not include spacy and torchvision package when "
            "using PytorchModelArtifact. To make sure BentoML bundle those packages if "
            "they are required for your model, either import those packages in "
            "BentoService definition file or manually add them via "
            "`@env(pip_packages=['torchvision'])` when defining a BentoService"
        )
        if env._infer_pip_packages:
            env.add_pip_packages(['torch'])

    def get(self):
        return self._model

    def save(self, dst):
        try:
            import torch
        except ImportError:
            raise MissingDependencyException(
                "torch package is required to use PytorchModelArtifact"
            )

        # If model is a TorchScriptModule, we cannot apply standard pickling
        if isinstance(self._model, torch.jit.ScriptModule):
            return torch.jit.save(self._model, self._file_path(dst))

        return cloudpickle.dump(self._model, open(self._file_path(dst), "wb"))


if __name__ == '__main__':
    pass
