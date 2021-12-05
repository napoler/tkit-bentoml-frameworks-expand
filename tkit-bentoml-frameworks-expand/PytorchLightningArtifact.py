import logging
import os
import pathlib
import shutil
"""

优化PytorchLightningModelArtifact



"""
from bentoml.exceptions import (
    InvalidArgument,
    MissingDependencyException,
)
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv

logger = logging.getLogger(__name__)


def _is_path_like(path):
    return isinstance(path, (str, bytes, pathlib.Path, os.PathLike))


def _is_pytorch_lightning_model_file_like(path):
    return (
        _is_path_like(path)
        and os.path.isfile(path)
        and str(path).lower().endswith(".pt")
    )



class PytorchLightningModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving and loading pytorch lightning model

    Args:
        name (string): Name of the pytorch model
    Raises:
        MissingDependencyException: torch and pytorch_lightning package is required.

    Example usage:

    """

    def __init__(self, name):
        super().__init__(name)
        self._model = None
        self._model_path = None

    def _saved_model_file_path(self, base_path):
        return os.path.join(base_path, self.name + '.pt')

    def pack(self, path_or_model, metadata=None):  # pylint:disable=arguments-renamed
        if _is_pytorch_lightning_model_file_like(path_or_model):
            logger.info(
                'PytorchLightningArtifact is packing a saved torchscript module '
                'from path'
            )
            self._model_path = path_or_model
        else:
            try:
                from pytorch_lightning.core.lightning import LightningModule
            except ImportError:
                raise InvalidArgument(
                    '"pytorch_lightning.lightning.LightningModule" model is required '
                    'to pack a PytorchLightningModelArtifact'
                )
            if isinstance(path_or_model, LightningModule):
                logger.info(
                    'PytorchLightningArtifact is packing a pytorch lightning '
                    'model instance as torchscript module'
                )
                self._model = path_or_model.to_torchscript()
            else:
                raise InvalidArgument(
                    'a LightningModule model is required to pack a '
                    'PytorchLightningModelArtifact'
                )
        return self

    def load(self, path):
        self._model = self._get_torch_script_model(self._saved_model_file_path(path))

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(['pytorch-lightning'])

    def get(self):
        if self._model is None:
            self._model = self._get_torch_script_model(self._model_path)
        return self._model

    def save(self, dst):
        if self._model:
            try:
                import torch
            except ImportError:
                raise MissingDependencyException(
                    '"torch" package is required for saving Pytorch lightning model'
                )
            torch.jit.save(self._model, self._saved_model_file_path(dst))
        if self._model_path:
            shutil.copyfile(self._model_path, self._saved_model_file_path(dst))

    @staticmethod
    def _get_torch_script_model(model_path):
        try:
            from torch import jit
        except ImportError:
            raise MissingDependencyException(
                '"torch" package is required for inference with '
                'PytorchLightningModelArtifact'
            )
        return jit.load(model_path)