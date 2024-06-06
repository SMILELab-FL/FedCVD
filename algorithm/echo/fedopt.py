
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


class FedOptServerHandler(FedAvgServerHandler):
    def __init__(
        self,
        lr: float,
        beta1: float,
        beta2: float,
        tau: float,
        option: str,
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
        super(FedOptServerHandler, self).__init__(
            num_classes=num_classes,
            model=model,
            test_loaders=test_loaders,
            criterion=criterion,
            output_path=output_path,
            evaluator=evaluator,
            communication_round=communication_round,
            num_clients=num_clients,
            sample_ratio=sample_ratio,
            device=device,
            logger=logger
        )
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.option = option
        self.momentum = torch.zeros_like(self.model_parameters)
        self.vt = torch.zeros_like(self.model_parameters)
        assert self.option in ["adagrad", "yogi", "adam"]

    def global_update(self, buffer):
        gradient_list = [
            torch.sub(ele[0], self.model_parameters) for ele in buffer
        ]
        delta = Aggregators.fedavg_aggregate(gradient_list, None)
        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * delta

        delta_2 = torch.pow(delta, 2)
        if self.option == "adagrad":
            self.vt += delta_2
        elif self.option == "yogi":
            self.vt = self.vt - (
                1 - self.beta2) * delta_2 * torch.sign(self.vt - delta_2)
        else:
            # adam
            self.vt = self.beta2 * self.vt + (1 - self.beta2) * delta_2

        serialized_parameters = self.model_parameters + self.lr * self.momentum / (
            torch.sqrt(self.vt) + self.tau)
        self.set_model(serialized_parameters)


class FedOptSerialClientTrainer(FedAvgSerialClientTrainer):
    def local_process(self, payload, id_list):
        pack = None
        model_parameters = payload[0]
        for idx in id_list:
            self.set_model(model_parameters)
            for epoch in range(self.max_epoch):
                pack = self.train(epoch, idx)
                self.local_test(idx, epoch)
                self.global_test(idx, epoch)
            self.cache.append(pack)
            torch.save(
                {
                    "model": self._model.state_dict()
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
            data, label = data.to(self._device), label.to(self._device)

            pred_score = self._model(data)

            loss = self.criterion(pred_score, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                pred_label = pred_score.argmax(dim=1)
                shield_pred_label = shield(pred_label, mask)
                micro_dice = self.dice_micro(shield_pred_label, label)
                macro_dice = self.dice_macro(shield_pred_label, label)
                hd = cal_hd(shield_pred_label.detach().cpu().numpy(), label.detach().cpu().numpy(), mask)
            metric.add(
                float(loss) * len(label),
                micro_dice * len(label),
                macro_dice * len(label),
                hd * len(label),
                len(label)
            )
            train_bar.desc = train_desc.format(
                epoch, metric[0] / metric[-1], metric[2] / metric[-1]
            )
            train_bar.update(1)
        train_bar.close()
        metric_dict = {
            "loss": metric[0] / metric[-1],
            "micro_dice": metric[1] / metric[-1],
            "macro_dice": metric[2] / metric[-1],
            "hd": metric[3] / metric[-1]
        }
        wandb.log(
            {
                f"client{idx + 1}_train_loss": metric[0] / metric[-1],
                f"client{idx + 1}_train_micro_dice": metric[1] / metric[-1],
                f"client{idx + 1}_train_macro_dice": metric[2] / metric[-1],
                f"client{idx + 1}_train_hd": metric[3] / metric[-1]
            }
        )
        self.evaluators[idx].add_dict("train", self.current_round, epoch, metric_dict)
        self._LOGGER[idx].info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[-1]} | Train Dice: {metric[2] / metric[-1]}")
        return [self.model_parameters]
