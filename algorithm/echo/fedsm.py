
from copy import deepcopy
from typing import List

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


class FedSMServerHandler(FedAvgServerHandler):
    def __init__(
        self,
        lambda_,
        gamma,
        num_classes: int,
        model: torch.nn.Module,
        model_selector: torch.nn.Module,
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
        super(FedSMServerHandler, self).__init__(
            num_classes=num_classes, model=model, test_loaders=test_loaders, criterion=criterion, output_path=output_path,
            evaluator=evaluator, communication_round=communication_round,
            num_clients=num_clients, sample_ratio=sample_ratio, device=device,
            logger=logger)
        self.num_classes = num_classes
        self.lambda_ = lambda_
        self.gamma = gamma
        self.local_models = [deepcopy(self._model).to(self._device) for _ in range(self.num_clients)]
        self.model_selector = deepcopy(model_selector).to(self._device)

    @property
    def downlink_package(self) -> List[torch.Tensor]:
        """Property for manager layer. Server manager will call this property when activates clients."""
        return [
            self.model_parameters,
            *[SerializationTool.serialize_model(ele) for ele in self.local_models],
            SerializationTool.serialize_model(self.model_selector)
        ]

    def global_update(self, buffer):
        g_parameters_list = [ele[0] for ele in buffer]
        l_parameters_list = [ele[1] for ele in buffer]
        s_parameters_list = [ele[2] for ele in buffer]
        weights = [ele[3] for ele in buffer]
        g_serialized_parameters = Aggregators.fedavg_aggregate(g_parameters_list, weights)
        s_serialized_parameters = Aggregators.fedavg_aggregate(s_parameters_list, weights)
        l_serialized_parameters = []
        for idx in range(self.num_clients):
            l_weights = [(1 - self.lambda_) / (self.num_clients - 1) for _ in range(self.num_clients)]
            l_weights[idx] = self.lambda_
            l_serialized_parameters.append(Aggregators.fedavg_aggregate(l_parameters_list, l_weights))
        SerializationTool.deserialize_model(self._model, g_serialized_parameters)
        SerializationTool.deserialize_model(self.model_selector, s_serialized_parameters)
        for idx in range(self.num_clients):
            SerializationTool.deserialize_model(self.local_models[idx], l_serialized_parameters[idx])

    def save_model(self, path):
        torch.save({
            "model_selector": self.model_selector.state_dict(),
            "model": self._model.state_dict()
        }, path)

    def local_test(self):
        self._model.eval()
        self.model_selector.eval()
        for idx in range(self.num_clients):
            self.local_models[idx].eval()
        l_metric_dict = {}
        eval_desc = "Local Test Loss {:.8f} | Dice {:.2f} | HD {:.2f}"
        for idx, item in enumerate(self.test_loaders):
            metric = Accumulator(5)
            eval_bar = tqdm.tqdm(initial=0, leave=True, total=len(item),
                                 desc=eval_desc.format(0, 0, 0), position=0)
            for data, label, mask in item:
                data, label = data.to(self._device), label.to(self._device)
                with torch.no_grad():
                    select_score = self.model_selector(data)
                    max_select_score, argmax_select_score = torch.max(select_score, dim=1)
                    argmax_select_score = torch.where(
                        max_select_score > self.gamma, argmax_select_score, -1
                    )
                    global_batch = torch.eq(argmax_select_score, -1).flatten()
                    local_batch = [torch.eq(argmax_select_score, i).flatten() for i in range(self.num_clients)]
                    pred_score = torch.zeros((label.shape[0], self.num_classes, *label.shape[1:])).to(self._device)
                    pred_score[global_batch] = self._model(data[global_batch])
                    for i in range(self.num_clients):
                        if torch.sum(local_batch[i]) > 0:
                            pred_score[local_batch[i]] = self.local_models[i](data[local_batch[i]])

                    pred_label = pred_score.argmax(dim=1)
                    shield_pred_label = shield(pred_label, mask)
                    micro_dice = self.dice_micro(shield_pred_label, label)
                    macro_dice = self.dice_macro(shield_pred_label, label)
                    hd = cal_hd(shield_pred_label.detach().cpu().numpy(), label.detach().cpu().numpy(), mask)

                    loss = self.criterion(pred_score, label)

                    metric.add(
                        float(loss) * len(label),
                        micro_dice * len(label),
                        macro_dice * len(label),
                        hd * len(label),
                        len(label)
                    )

                eval_bar.desc = eval_desc.format(
                    metric[0] / metric[-1], metric[2] / metric[-1], metric[3] / metric[-1]
                )
                eval_bar.update(1)
            eval_bar.close()
            metric_dict = {
                "loss": metric[0] / metric[-1],
                "micro_dice": metric[1] / metric[-1],
                "macro_dice": metric[2] / metric[-1],
                "hd": metric[3] / metric[-1]
            }
            wandb.log(
                {
                    f"client{idx + 1}_local_test_loss": metric[0] / metric[-1],
                    f"client{idx + 1}_test_micro_dice": metric[1] / metric[-1],
                    f"client{idx + 1}_test_macro_dice": metric[2] / metric[-1],
                    f"client{idx + 1}_test_hd": metric[3] / metric[-1]
                }
            )
            l_metric_dict[str(idx)] = metric_dict
            self._LOGGER.info(f"Round {self.round} | Client {idx + 1} Local Test Loss: {metric[0] / metric[-1]} | Local Test Dice: {metric[2] / metric[-1]}")
        self.evaluator.add_dict("local_test", self.round, l_metric_dict)

    def global_test(self):
        self._model.eval()
        self.model_selector.eval()
        for idx in range(self.num_clients):
            self.local_models[idx].eval()
        metric = Accumulator(5)
        eval_desc = " Global Test Loss {:.8f} | Dice {:.2f} | HD {:.2f}"
        length = 0
        for item in self.test_loaders:
            length += len(item)
        eval_bar = tqdm.tqdm(initial=0, leave=True, total=length,
                        desc=eval_desc.format(0, 0, 0), position=0)
        for item in self.test_loaders:
            for data, label, mask in item:
                data, label = data.to(self._device), label.to(self._device)
                with torch.no_grad():
                    select_score = self.model_selector(data)
                    max_select_score, argmax_select_score = torch.max(select_score, dim=1)
                    argmax_select_score = torch.where(
                        max_select_score > self.gamma, argmax_select_score, -1
                    )
                    global_batch = torch.eq(argmax_select_score, -1).flatten()
                    local_batch = [torch.eq(argmax_select_score, i).flatten() for i in range(self.num_clients)]
                    pred_score = torch.zeros((label.shape[0], self.num_classes, *label.shape[1:])).to(self._device)
                    pred_score[global_batch] = self._model(data[global_batch])
                    for i in range(self.num_clients):
                        if torch.sum(local_batch[i]) > 0:
                            pred_score[local_batch[i]] = self.local_models[i](data[local_batch[i]])

                    pred_label = pred_score.argmax(dim=1)
                    shield_pred_label = shield(pred_label, mask)
                    micro_dice = self.dice_micro(shield_pred_label, label)
                    macro_dice = self.dice_macro(shield_pred_label, label)
                    hd = cal_hd(shield_pred_label.detach().cpu().numpy(), label.detach().cpu().numpy(), mask)

                    loss = self.criterion(pred_score, label)

                    metric.add(
                        float(loss) * len(label),
                        micro_dice * len(label),
                        macro_dice * len(label),
                        hd * len(label),
                        len(label)
                    )

                eval_bar.desc = eval_desc.format(metric[0] / metric[-1], metric[2] / metric[-1], metric[3] / metric[-1])
                eval_bar.update(1)
        eval_bar.close()
        metric_dict = {
            "loss": metric[0] / metric[-1],
            "micro_dice": metric[1] / metric[-1],
            "macro_dice": metric[2] / metric[-1],
            "hd": metric[3] / metric[-1]
        }
        wandb.log(
            {
                "global_test_loss": metric[0] / metric[-1],
                "global_test_micro_dice": metric[1] / metric[-1],
                "global_test_macro_dice": metric[2] / metric[-1],
                "global_test_hd": metric[3] / metric[-1]
            }
        )
        self.evaluator.add_dict("global_test", self.round, metric_dict)
        self._LOGGER.info(f"Round {self.round} | Global Test Loss: {metric[0] / metric[-1]} | Global Test Dice: {metric[2] / metric[-1]}")


class FedSMSerialClientTrainer(FedAvgSerialClientTrainer):
    def __init__(
            self,
            model,
            model_selector,
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
        super(FedSMSerialClientTrainer, self).__init__(
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
        self.model_selector = deepcopy(model_selector).to(self._device)
        self.local_model = deepcopy(model).to(self._device)
        self.global_models = []
        self.local_models = [self.model_parameters for _ in range(self.num_clients)]
        self.s_criterion = torch.nn.CrossEntropyLoss()
        if optimizer_name == "SGD":
            self.s_optimizer = torch.optim.SGD(self.model_selector.parameters(), lr=lr)
        elif optimizer_name == "Adam":
            self.s_optimizer = torch.optim.Adam(self.model_selector.parameters(), lr=lr)
        else:
            raise ValueError("Invalid optimizer name")
        # self.s_optimizer = torch.optim.SGD(self.model_selector.parameters(), lr=lr)

    def finish(self):
        for idx in range(self.num_clients):
            self.evaluators[idx].save(self.output_path + "client" + str(idx + 1) + "/metric.json")
            self._LOGGER[idx].close()

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for idx in id_list:
            pack = None
            global_update, local_update, selector_update = None, None, None
            self.set_model(model_parameters)
            # SerializationTool.deserialize_model(self.local_model, payload[idx + 1])
            # SerializationTool.deserialize_model(self.model_selector, payload[-1])
            # for epoch in range(self.max_epoch):
            #     pack = self.train(epoch, idx)
            # self.local_models[idx] = pack[1]
            for epoch in range(self.max_epoch):
                global_update = self.train_global_model(epoch, idx)
            self.set_model(payload[idx + 1])
            for epoch in range(self.max_epoch):
                local_update = self.train_local_model(epoch, idx)
            self.local_models[idx] = local_update[0]
            SerializationTool.deserialize_model(self.model_selector, payload[-1])
            for epoch in range(self.max_epoch):
                selector_update = self.train_selector(epoch, idx)
            pack = [global_update[0], local_update[0], selector_update[0], global_update[1]]
            self.cache.append(pack)
            torch.save(
                {
                    "model_selector": self.model_selector.state_dict(),
                    "model": self._model.state_dict(),
                },
                self.output_path + "client" + str(idx + 1) + "/model.pth"
            )

    def train(self, epoch, idx):
        self._model.train()
        self.local_model.train()
        self.model_selector.train()
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
            loc = torch.tensor([idx for _ in range(len(data))], dtype=torch.long)
            data, label, loc = data.to(self._device), label.to(self._device), loc.to(self._device)
            pred_score = self._model(data)

            # loss = self.criterion(pred_score, train_label)
            g_loss = self.criterion(pred_score, label)
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

            l_loss = self.criterion(self.local_model(data), label)
            self.optimizer.zero_grad()
            l_loss.backward()
            self.optimizer.step()

            s_loss = self.s_criterion(self.model_selector(data), loc)
            self.s_optimizer.zero_grad()
            s_loss.backward()
            self.s_optimizer.step()

            # with torch.no_grad():
            #     pred_label = pred_score.argmax(dim=1)
            #     shield_pred_label = shield(pred_label, mask)
            #     micro_dice = self.dice_micro(shield_pred_label, label)
            #     macro_dice = self.dice_macro(shield_pred_label, label)
            #     hd = cal_hd(shield_pred_label.detach().cpu().numpy(), label.detach().cpu().numpy(), mask)
            metric.add(
                float(g_loss) * len(label), 0, 0, 0,
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
        return [
            self.model_parameters,
            SerializationTool.serialize_model(self.local_model),
            SerializationTool.serialize_model(self.model_selector),
            metric[-1]
        ]

    def train_global_model(self, epoch, idx):
        self._model.train()
        metric = Accumulator(5)
        train_desc = "Epoch {:2d}: train Loss {:.8f} | Dice {:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loaders[idx]),
                         desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label, mask in self.train_loaders[idx]:
            # train_label = deepcopy(label)
            unlabeled_batch = torch.ne(mask, 0).flatten()
            # train_label[unlabeled_batch] = torch.where(train_label[unlabeled_batch] == 0, -1,
                                                       # train_label[unlabeled_batch])
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

    def train_local_model(self, epoch, idx):
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

    def train_selector(self, epoch, idx):
        self.model_selector.train()

        metric = Accumulator(2)
        train_desc = "Epoch {:2d}: train Loss {:.8f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loaders[idx]),
                              desc=train_desc.format(epoch, 0, 0), position=0)
        for data, _, _ in self.train_loaders[idx]:
            label = torch.tensor([idx for _ in range(len(data))], dtype=torch.long)
            data, label = data.to(self._device), label.to(self._device)

            pred_score = self.model_selector(data)

            loss = self.s_criterion(pred_score, label)

            self.s_optimizer.zero_grad()
            loss.backward()
            self.s_optimizer.step()

            metric.add(
                float(loss) * len(label), len(label)
            )
            train_bar.desc = train_desc.format(epoch, metric[0] / metric[-1])
            train_bar.update(1)
        train_bar.close()
        self._LOGGER[idx].info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[-1]}")

        return [SerializationTool.serialize_model(self.model_selector), metric[-1]]
