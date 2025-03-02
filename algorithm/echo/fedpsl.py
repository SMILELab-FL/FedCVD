
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


class FedPSLServerHandler(FedAvgServerHandler):
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        c_weights = [ele[2] for ele in buffer]
        print(c_weights)
        c_idx = SerializationTool.serialize_model(self._model.out).size(0)
        weight_step = (c_idx - self.num_classes) // self.num_classes
        bias_step = 1
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        for idx in range(self.num_classes):
            serialized_parameters[-c_idx: -c_idx + weight_step] = Aggregators.fedavg_aggregate(
                [ele[-c_idx: -c_idx + weight_step] for ele in parameters_list],
                [ele[idx] for ele in c_weights]
            )
            c_idx -= weight_step
        for idx in range(self.num_classes):
            if -c_idx + bias_step != 0:
                serialized_parameters[-c_idx: -c_idx + bias_step] = Aggregators.fedavg_aggregate(
                    [ele[-c_idx: -c_idx + bias_step] for ele in parameters_list],
                    [ele[idx] for ele in c_weights]
                )
                print(serialized_parameters[-c_idx: -c_idx + bias_step],
                      [ele[-c_idx: -c_idx + bias_step] for ele in parameters_list],
                      [ele[idx] for ele in c_weights])
            else:
                serialized_parameters[-c_idx:] = Aggregators.fedavg_aggregate(
                    [ele[-c_idx:] for ele in parameters_list],
                    [ele[idx] for ele in c_weights]
                )
                print(serialized_parameters[-c_idx:],
                      [ele[-c_idx:] for ele in parameters_list],
                      [ele[idx] for ele in c_weights])
            c_idx -= bias_step
        assert c_idx == 0
        SerializationTool.deserialize_model(self._model, serialized_parameters)


class FedPSLSerialClientTrainer(FedAvgSerialClientTrainer):
    def __init__(
            self,
            alpha,
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
        super(FedPSLSerialClientTrainer, self).__init__(
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
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        if optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.alpha)
            self.meta_optimizer = torch.optim.SGD(self._model.out.parameters(), lr=self.beta)
        elif optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.alpha)
            self.meta_optimizer = torch.optim.Adam(self._model.out.parameters(), lr=self.beta)

    def finish(self):
        for idx in range(self.num_clients):
            self.evaluators[idx].save(self.output_path + "client" + str(idx + 1) + "/metric.json")
            self._LOGGER[idx].close()

    def local_process(self, payload, id_list):
        pack = None
        model_parameters = payload[0]
        for idx in id_list:
            self.set_model(model_parameters)
            for epoch in range(self.max_epoch):
                pack = self.train(epoch, idx)
                # self.local_test(idx, epoch)
                # self.global_test(idx, epoch)
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
        class_num = Accumulator(4)
        train_desc = "Epoch {:2d}: train Loss {:.8f} | Dice {:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loaders[idx]),
                         desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label, mask in self.train_loaders[idx]:
            val_idx = data.shape[0] // 2
            unlabeled_batch = torch.ne(mask, 0).flatten()
            label[unlabeled_batch] = torch.where(label[unlabeled_batch] == 0, -1, label[unlabeled_batch])
            data, label = data.to(self._device), label.to(self._device)
            classifier_parameters = deepcopy(list(self._model.out.parameters()))
            pred_score = self._model(data[:val_idx])

            # loss = self.criterion(pred_score, train_label)
            loss = self.criterion(pred_score, label[:val_idx])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_score = self._model(data[val_idx:])
            meta_loss = self.criterion(pred_score, label[val_idx:])
            for param in self._model.out.parameters():
                param.data.copy_(classifier_parameters.pop(0).data)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
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
            class_num.add(
                torch.sum(mask == 0).item(),
                torch.sum(mask == 0).item() + torch.sum(mask == 1).item(),
                torch.sum(mask == 0).item() + torch.sum(mask == 2).item(),
                torch.sum(mask == 0).item() + torch.sum(mask == 3).item(),
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
        assert metric[-1] in (class_num[0], class_num[1], class_num[2], class_num[3])
        self._LOGGER[idx].info(f"({class_num[0]}, {class_num[1]}, {class_num[2]}, {class_num[3]})")
        return [self.model_parameters, metric[-1], (class_num[0], class_num[1], class_num[2], class_num[3])]
