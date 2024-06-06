import torch
import tqdm
from tqdm import *
from copy import deepcopy

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from ...utils.serialization import SerializationTool


##################
#
#      Server
#
##################


class DittoServerHandler(SyncServerHandler):
    """Ditto server acts the same as fedavg server."""
    None


##################
#
#      Client
#
##################


class DittoSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num, cuda=False, device=None, logger=None, personal=True) -> None:
        super().__init__(model, num, cuda, device, logger, personal)
        self.ditto_gmodels = []
        self.local_models = self.parameters
    
    def setup_dataset(self, dataset):
        return super().setup_dataset(dataset)
    
    def setup_optim(self, epochs, batch_size, lr):
        return super().setup_optim(epochs, batch_size, lr)

    def local_process(self, payload, id_list):
        global_model = payload[0]
        for id in tqdm(id_list):
            # self._LOGGER.info("Local process is running. Training client {}".format(id))
            train_loader = self.dataset.get_dataloader(id, batch_size=self.batch_size)
            self.local_models[id], glb_model  = self.train(global_model, self.local_models[id], train_loader)
            self.ditto_gmodels.append(deepcopy(glb_model))

    @property
    def uplink_package(self):
        ditto_gmodels = deepcopy(self.ditto_gmodels)
        self.ditto_gmodels = []
        return [[parameter] for parameter in ditto_gmodels]

    def train(self, global_model_parameters, local_model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, global_model_parameters)
        self._model.train()
        for ep in range(self.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.device), label.cuda(self.device)

                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        updated_glb_models = deepcopy(self.model_parameters)

        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, global_model_parameters)

        SerializationTool.deserialize_model(self._model, local_model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr)

        self._model.train()
        for ep in range(self.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.device), label.cuda(self.device)

                preds = self._model(data)
                l1 = criterion(preds,label)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                # loss = l1 + 0.5 * self.args.mu * l2
                loss = l1 + 0.5 * 0.1 * l2  # fedprox 的 mu
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.model_parameters, updated_glb_models
