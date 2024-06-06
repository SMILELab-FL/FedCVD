
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

import torch
from torch import nn
import random
from copy import deepcopy

from typing import List
from ..utils import Logger, Aggregators, SerializationTool
from ..utils.functional import evaluate
from ..core.server.handler import ServerHandler


class SyncServerHandler(ServerHandler):
    """Synchronous Parameter Server Handler.

    Backend of synchronous parameter server: this class is responsible for backend computing in synchronous server.

    Synchronous parameter server will wait for every client to finish local training process before
    the next FL round.

    Details in paper: http://proceedings.mlr.press/v54/mcmahan17a.html

    Args:
        model (torch.nn.Module): model trained by federated learning.
        communication_round (int): stop condition. Shut down FL system when global round is reached.
        num_clients (int): number of clients in FL. Default: 0 (initialized external).
        sample_ratio (float): the result of ``sample_ratio * num_clients`` is the number of clients for every FL round.
        device (str, optional): assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
        logger (Logger, optional): object of :class:`Logger`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loaders,
        criterion: nn.Module,
        output_path: str,
        evaluator,
        communication_round: int,
        num_clients: int = 0,
        sample_ratio: float = 1,
        device: torch.device | None = None,
        logger: Logger = None,
    ):
        super(SyncServerHandler, self).__init__(model, device)
        self.test_loaders = test_loaders
        self.criterion = criterion
        self.output_path = output_path
        self.evaluator = evaluator

        self._LOGGER = Logger() if logger is None else logger
        assert 0.0 <= sample_ratio <= 1.0

        # basic setting
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio

        # client buffer
        self.round_clients = max(
            1, int(self.sample_ratio * self.num_clients)
        )  # for dynamic client sampling
        self.client_buffer_cache = []

        # stop condition
        self.communication_round = communication_round
        self.round = 0

    @property
    def downlink_package(self) -> List[torch.Tensor]:
        """Property for manager layer. Server manager will call this property when activates clients."""
        return [self.model_parameters]

    @property
    def num_clients_per_round(self):
        return self.round_clients

    @property
    def if_stop(self):
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return self.round >= self.communication_round

    # for built-in sampler
    # @property
    # def num_clients_per_round(self):
    #     return max(1, int(self.sample_ratio * self.num_clients))

    # def setup_optim(self, num_clients):
    #     self.num_clients = num_clients

    def sample_clients(self, num_to_sample=None):
        """Return a list of client rank indices selected randomly. The client ID is from ``0`` to
        ``self.num_clients -1``."""
        # selection = random.sample(range(self.num_clients),
        #                           self.num_clients_per_round)
        # If the number of clients per round is not fixed, please change the value of self.sample_ratio correspondly.
        # self.sample_ratio = float(len(selection))/self.num_clients
        # assert self.num_clients_per_round == len(selection)

        # new version with built-in sampler
        sampled = random.sample(range(self.num_clients), self.round_clients)
        assert self.num_clients_per_round == len(sampled)
        return sorted(sampled)

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

    def load(self, payload: List[torch.Tensor]) -> bool:
        """Update global model with collected parameters from clients.

        Note:
            Server handler will call this method when its ``client_buffer_cache`` is full. User can
            overwrite the strategy of aggregation to apply on :attr:`model_parameters_list`, and
            use :meth:`SerializationTool.deserialize_model` to load serialized parameters after
            aggregation into :attr:`self._model`.

        Args:
            payload (list[torch.Tensor]): A list of tensors passed by manager layer.
        """
        assert len(payload) > 0
        self.client_buffer_cache.append(deepcopy(payload))

        assert len(self.client_buffer_cache) <= self.num_clients_per_round

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            # self.round += 1

            # reset cache
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False

    def save_model(self, path):
        torch.save({
            "model": self._model.state_dict()
        }, path)

    def evaluate(self, evaluator):
        raise NotImplementedError("Please implement the evaluation function.")
