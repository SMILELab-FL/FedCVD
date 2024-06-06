
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


class ScaffoldServerHandler(FedAvgServerHandler):
    def __init__(
            self,
            lr: float,
            num_classes: int,
            model: torch.nn.Module,
            test_loaders,
            criterion: torch.nn.Module,
            output_path: str,
            evaluator,
            communication_round: int,
            num_clients: int = 0,
            sample_ratio: float = 1,
            device: torch.device | None = None,
            logger: Logger = None,
    ):
        super(ScaffoldServerHandler, self).__init__(
            num_classes=num_classes,
            model=model,
            test_loaders=test_loaders,
            criterion=criterion,
            output_path=output_path,
            evaluator=evaluator,
            communication_round=communication_round,
            num_clients= num_clients,
            sample_ratio=sample_ratio,
            device=device,
            logger=logger
        )
        self.lr = lr
        self.global_c = torch.zeros_like(self.model_parameters)
    @property
    def downlink_package(self):
        return [self.model_parameters, self.global_c]

    def global_update(self, buffer):
        # unpack
        dys = [ele[0] for ele in buffer]
        dcs = [ele[1] for ele in buffer]

        dx = Aggregators.fedavg_aggregate(dys)
        dc = Aggregators.fedavg_aggregate(dcs)

        next_model = self.model_parameters + self.lr * dx
        self.set_model(next_model)

        self.global_c += 1.0 * len(dcs) / self.num_clients * dc

class ScaffoldSerialClientTrainer(FedAvgSerialClientTrainer):
    def __init__(
            self,
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
            device: torch.device | None = None,
            logger=None,
            personal=False
    ):
        super(ScaffoldSerialClientTrainer, self).__init__(
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
        self.cs = [None for _ in range(self.num_clients)]

    def finish(self):
        for idx in range(self.num_clients):
            self.evaluators[idx].save(self.output_path + "client" + str(idx + 1) + "/metric.json")
            self._LOGGER[idx].close()

    def local_process(self, payload, id_list):
        pack = None
        model_parameters = payload[0]
        global_c = payload[1]
        for idx in id_list:
            self.set_model(model_parameters)
            frz_model = model_parameters
            if self.cs[idx] is None:
                self.cs[idx] = torch.zeros_like(model_parameters)
            for epoch in range(self.max_epoch):
                pack = self.train(epoch, global_c, idx)
                # self.local_test(idx, epoch)
                # self.global_test(idx, epoch)
            dy = self.model_parameters - frz_model
            dc = -1.0 / (self.max_epoch * len(self.train_loaders[idx]) * self.lr) * dy - global_c
            self.cs[idx] += dc
            self.cache.append([dy, dc])
            torch.save(
                {
                    "model": self._model.state_dict()
                },
                self.output_path + "client" + str(idx + 1) + "/model.pth"
            )

    def train(self, epoch, global_c, idx):
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
            label[unlabeled_batch] = torch.where(label[unlabeled_batch] == 0, -1, label[unlabeled_batch])
            data, label = data.to(self._device), label.to(self._device)
            # data, train_label, label = data.to(self._device), train_label.to(self._device), label.to(self._device)

            pred_score = self._model(data)

            loss = self.criterion(pred_score, label)
            # loss = self.criterion(pred_score, train_label)

            self.optimizer.zero_grad()
            loss.backward()

            grad = self.model_grads
            grad = grad - self.cs[idx] + global_c
            index = 0

            parameters = self._model.parameters()
            for p in self._model.state_dict().values():
                if p.grad is None:
                    layer_size = p.numel()
                else:
                    parameter = next(parameters)
                    layer_size = parameter.data.numel()
                    shape = parameter.grad.shape
                    parameter.grad.data[:] = grad[index:index + layer_size].view(shape)[:]
                index += layer_size


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
