
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from algorithm.echo.fedavg import FedAvgServerHandler, FedAvgSerialClientTrainer
from fedlab.utils import Aggregators
from fedlab.utils import SerializationTool
from fedlab.algorithm.base_client import SGDSerialClientTrainer
from fedlab.algorithm.base_server import SyncServerHandler
from utils.evaluation import Accumulator
from utils.evaluation import shield, cal_hd
from utils.io import guarantee_path
from fedlab.utils.logger import Logger
from torchmetrics import Dice

import random
import torch
import numpy as np
import tqdm
import pandas as pd
import wandb


class FedALAServerHandler(FedAvgServerHandler):
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


class FedALASerialClientTrainer(FedAvgSerialClientTrainer):
    def __init__(
            self,
            rand_percent: int,
            model,
            num_clients,
            batch_size,
            train_datasets,
            train_loaders,
            test_loaders,
            num_classes: int,
            lr: float,
            criterion: torch.nn.Module,
            max_epoch: int,
            output_path: str,
            evaluators,
            layer_idx: int = 0,
            eta: float = 1.0,
            threshold: float = 0.1,
            num_pre_loss: int = 10,
            device: torch.device | None = None,
            logger=None,
            personal=False
    ):
        super(FedALASerialClientTrainer, self).__init__(
            model=model,
            num_clients=num_clients,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_classes=num_classes,
            lr=lr,
            criterion=criterion,
            max_epoch=max_epoch,
            output_path=output_path,
            evaluators=evaluators,
            device=device,
            logger=logger,
            personal=personal
        )
        self.batch_size = batch_size
        self.train_datasets = train_datasets
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.global_models = []
        self.local_models = [self.model_parameters for _ in range(self.num_clients)]
        self.weights = [None for _ in range(self.num_clients)]
        self.start_phase = [True for _ in range(self.num_clients)]

    def finish(self):
        for idx in range(self.num_clients):
            self.evaluators[idx].save(self.output_path + "client" + str(idx + 1) + "/metric.json")
            self._LOGGER[idx].close()

    def local_process(self, payload, id_list):
        pack = None
        global_parameters = payload[0]
        for idx in id_list:
            local_parameters = self.adaptive_local_aggregation(global_parameters, self.local_models[idx], idx)
            self.set_model(local_parameters)
            for epoch in range(self.max_epoch):
                pack = self.train(epoch, idx)
                self.local_test(idx, epoch)
                self.global_test(idx, epoch)
            self.local_models[idx] = self.model_parameters
            self.cache.append(pack)
            assert torch.equal(pack[0], self.local_models[idx])
            torch.save(
                {
                    "model": self._model.state_dict(),
                },
                self.output_path + "client" + str(idx + 1) + "/model.pth"
            )

    def train(self, epoch, idx):
        self._model.train()
        metric = Accumulator(5)
        train_desc = "Epoch {:2d}: train Loss {:.8f} | Dice {:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loaders[idx]),
                         desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label, mask in self.train_loaders[idx]:
            # train_label = deepcopy(label)
            unlabeled_batch = torch.ne(mask, 0).flatten()
            # train_label[unlabeled_batch] = torch.where(train_label[unlabeled_batch] == 0, -1,
            #                                            train_label[unlabeled_batch])
            # data, train_label, label = data.to(self._device), train_label.to(self._device), label.to(self._device)
            label[unlabeled_batch] = torch.where(label[unlabeled_batch] == 0, -1, label[unlabeled_batch])
            data, label = data.to(self._device), label.to(self._device)
            pred_score = self._model(data)

            # loss = self.criterion(pred_score, train_label)
            loss = self.criterion(pred_score, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # with torch.no_grad():
            #     pred_label = pred_score.argmax(dim=1)
            #     shield_pred_label = shield(pred_label, mask)
            #     micro_dice = self.dice_micro(shield_pred_label, label)
            #     macro_dice = self.dice_macro(shield_pred_label, label)
            #     hd = cal_hd(shield_pred_label.detach().cpu().numpy(), label.detach().cpu().numpy(), mask)
            metric.add(
                float(loss) * len(label), 0, 0, 0,
                # micro_dice * len(label),
                # macro_dice * len(label),
                # hd * len(label),
                len(label)
            )
            train_bar.desc = train_desc.format(
                epoch, metric[0] / metric[-1], metric[2] / metric[-1]
            )
            train_bar.update(1)
        train_bar.close()
        # metric_dict = {
        #     "loss": metric[0] / metric[-1],
        #     "micro_dice": metric[1] / metric[-1],
        #     "macro_dice": metric[2] / metric[-1],
        #     "hd": metric[3] / metric[-1]
        # }
        # wandb.log(
        #     {
        #         f"client{idx + 1}_train_loss": metric[0] / metric[-1],
        #         f"client{idx + 1}_train_micro_dice": metric[1] / metric[-1],
        #         f"client{idx + 1}_train_macro_dice": metric[2] / metric[-1],
        #         f"client{idx + 1}_train_hd": metric[3] / metric[-1]
        #     }
        # )
        # self.evaluators[idx].add_dict("train", self.current_round, epoch, metric_dict)
        # self._LOGGER[idx].info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[-1]} | Train Dice: {metric[2] / metric[-1]}")
        return [self.model_parameters, metric[-1]]

    def adaptive_local_aggregation(self,
                                   global_parameters,
                                   local_parameters,
                                   idx):
        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_datasets[idx]))
        rand_idx = random.randint(0, len(self.train_datasets[idx]) - rand_num)
        subset = Subset(self.train_datasets[idx], range(rand_idx, rand_idx + rand_num))
        rand_loader = DataLoader(
            subset, self.batch_size
        )

        # obtain the references of the parameters
        global_model = deepcopy(self._model)
        local_model = deepcopy(self._model)
        SerializationTool.deserialize_model(global_model, global_parameters)
        params_g = list(global_model.parameters())
        SerializationTool.deserialize_model(local_model, local_parameters)
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            assert torch.equal(SerializationTool.serialize_model(local_model), local_parameters)
            return SerializationTool.serialize_model(local_model)

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        model_t = deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights[idx] is None:
            self.weights[idx] = [torch.ones_like(param.data).to(self._device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                   self.weights[idx]):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            loss = 0
            num = 0
            for data, label, mask in rand_loader:
                unlabeled_batch = torch.ne(mask, 0).flatten()
                label[unlabeled_batch] = torch.where(label[unlabeled_batch] == 0, -1, label[unlabeled_batch])
                data, label = data.to(self._device), label.to(self._device)
                optimizer.zero_grad()
                output = model_t(data)
                loss_value = self.criterion(output, label)  # modify according to the local objective
                loss_value.backward()
                with torch.no_grad():
                    loss += loss_value.item() * len(label)
                    num += len(label)

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                           params_gp, self.weights[idx]):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                           params_gp, self.weights[idx]):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss / num)
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase[idx]:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                self._LOGGER[idx].info(
                    f'Client:{idx + 1}\tStd: {np.std(losses[-self.num_pre_loss:])}\tALA epochs: {cnt}'
                )
                break

        self.start_phase[idx] = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
        return SerializationTool.serialize_model(local_model)
