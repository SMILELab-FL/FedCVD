
from .base_server import SyncServerHandler
from .base_client import SGDClientTrainer, SGDSerialClientTrainer
from ..utils.aggregator import Aggregators
from ..utils.serialization import SerializationTool

##################
#
#      Server
#
##################


class FedAvgServerHandler(SyncServerHandler):
    """FedAvg server handler."""

    def setup_optim(self):
        pass

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

    def evaluate(self, evaluator):
        pass


##################
#
#      Client
#
##################


class FedAvgClientTrainer(SGDClientTrainer):
    """Federated client with local SGD solver."""
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(
            parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


class FedAvgSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self._model.train()

        data_size = 0
        for _ in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.cuda(self._device)
                target = target.cuda(self._device)

                output = self.model(data)
                loss = self.criterion(output, target)

                data_size += len(target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters, data_size]
