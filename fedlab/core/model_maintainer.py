# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import torch
from copy import deepcopy
from ..utils.serialization import SerializationTool
from ..utils.functional import get_best_gpu


class ModelMaintainer(object):
    """Maintain PyTorch model.

    Provide necessary attributes and operation methods. More features with local or global model
    will be implemented here.

    Args:
        model (torch.nn.Module): PyTorch model.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device | None = None
                 ) -> None:
        # dynamic device acquire.
        self._device = get_best_gpu() if device is None else device
        self._model = deepcopy(model).to(self._device)

    def set_model(self, parameters: torch.Tensor):
        """Assign parameters to self._model."""
        SerializationTool.deserialize_model(self._model, parameters)

    @property
    def model(self) -> torch.nn.Module:
        """Return :class:`torch.nn.module`."""
        return self._model

    @property
    def model_parameters(self) -> torch.Tensor:
        """Return serialized model parameters."""
        return SerializationTool.serialize_model(self._model)

    @property
    def model_grads(self) -> torch.Tensor: 
        """Return serialized model gradients(base on model.state_dict(), Shape is the same as model_parameters)."""
        params = self._model.state_dict()
        for name, p in self._model.named_parameters():
            params[name].grad = p.grad
        for key in params:
            if params[key].grad is None:
                params[key].grad = torch.zeros_like(params[key])
        gradients = [param.grad.data.view(-1) for param in params.values()]
        m_gradients = torch.cat(gradients)
        m_gradients = m_gradients.cpu()
        return m_gradients

    @property
    def model_gradients(self) -> torch.Tensor:
        """Return serialized model gradients."""
        return SerializationTool.serialize_model_gradients(self._model)

    @property
    def shape_list(self) -> List:
        """Return shape of model parameters.
        
        Currently, this attributes used in tensor compression.
        """
        shape_list = [param.shape for param in self._model.parameters()]
        return shape_list


class SerialModelMaintainer(ModelMaintainer):
    """"Maintain PyTorch model.

    Provide necessary attributes and operation methods. More features with local or global model
    will be implemented here.

    Args:
        model (torch.nn.Module): PyTorch model.
        num_clients (int): The number of independent models.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest idle memory as default.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 num_clients: int,
                 device: torch.device | None  = None,
                 personal: bool = False) -> None:
        super().__init__(model, device)
        self.parameters = [self.model_parameters for _ in range(num_clients)] if personal else None

    def set_model(self, parameters: torch.Tensor = None, idx: int = None):
        """Assign parameters to self._model.

        Note:
            parameters and id can not be None at the same time. 
            If id is None, this function load the given parameters.
            If id is not None, this function load the parameters of given id first and the parameters attribute will be ignored.

        Args:
            parameters (torch.Tensor, optional): Model parameters. Defaults to None.
            idx (int, optional): Load the model parameters of client id. Defaults to None.
        """
        if idx is None:
            super().set_model(parameters)
        else:
            super().set_model(self.parameters[idx])
