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

from copy import deepcopy
import torch
from tqdm import tqdm
from ..core.client.trainer import ClientTrainer, SerialClientTrainer
from ..utils import Logger, SerializationTool

class SGDClientTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): :object of :class:`Logger`.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader,
                 test_loader,
                 device: torch.device | None = None,
                 logger: Logger = None
                 ):
        super(SGDClientTrainer, self).__init__(model, device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        return [self.model_parameters]

    def setup_optim(self, epochs, lr, criterion):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size.
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = criterion

    def local_process(self, payload, idx):
        model_parameters = payload[0]
        self.train(model_parameters, self.train_loader)


    def train(self, model_parameters, train_loader) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                data, target = data.cuda(self._device), target.cuda(self._device)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")


class SGDSerialClientTrainer(SerialClientTrainer):
    """
    Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        num_clients (int): Number of clients in current trainer.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): Object of :class:`Logger`.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """
    def __init__(
            self, model, num_clients,
            train_loaders,
            test_loaders,
            device: torch.device | None=None, logger=None, personal=False
    ) -> None:
        super().__init__(model, num_clients, device, personal)
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self._LOGGER = Logger() if logger is None else logger
        self.cache = []

    def setup_optim(self, epochs, lr, criterion):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = criterion

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for idx in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {idx}", refresh=True)
            data_loader = self.train_loaders[idx]
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)

    def train(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                data = data.cuda(self._device)
                target = target.cuda(self._device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]