
from copy import deepcopy
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

import torch
import numpy as np
import tqdm
import pandas as pd
import wandb


class FedInitServerHandler(FedAvgServerHandler):
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, None)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


class FedInitSerialClientTrainer(FedAvgSerialClientTrainer):
    def __init__(
            self,
            beta,
            model,
            num_clients,
            train_loaders,
            test_loaders,
            num_classes: int,
            lr: float,
            criterion: torch.nn.Module,
            max_epoch: int,
            output_path: str,
            evaluators,
            optimizer_name: str = "SGD",
            device: torch.device | None = None,
            logger=None,
            personal=False
    ):
        super(FedInitSerialClientTrainer, self).__init__(
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
            optimizer_name=optimizer_name,
            device=device,
            logger=logger,
            personal=personal
        )
        self.beta = beta
        self.global_models = []
        self.local_models = [self.model_parameters for _ in range(self.num_clients)]

    def finish(self):
        for idx in range(self.num_clients):
            self.evaluators[idx].save(self.output_path + "client" + str(idx + 1) + "/metric.json")
            self._LOGGER[idx].close()

    def local_process(self, payload, id_list):
        pack = None
        global_model = payload[0]
        for idx in id_list:
            model_parameters = global_model + self.beta * (global_model - self.local_models[idx])
            self.set_model(model_parameters)
            for epoch in range(self.max_epoch):
                pack = self.train(epoch, idx)
                # self.local_test(idx, epoch)
                # self.global_test(idx, epoch)
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
        return [self.model_parameters]
