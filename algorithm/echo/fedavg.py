
from copy import deepcopy
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


class FedAvgServerHandler(SyncServerHandler):
    """FedAvg server handler."""
    def __init__(
        self,
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
        super(FedAvgServerHandler, self).__init__(model, test_loaders, criterion, output_path, evaluator, communication_round, num_clients, sample_ratio, device, logger)
        self.dice_macro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        self.dice_micro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        guarantee_path(self.output_path + "server/train/")
        guarantee_path(self.output_path + "server/local_test/")
        guarantee_path(self.output_path + "server/global_test/")

    def setup_optim(self):
        pass

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


    def finish(self):
        self.evaluator.save(self.output_path + "server/metric.json")
        self.save_model(self.output_path + "server/model.pth")

    def local_test(self):
        self._model.eval()
        l_metric_dict = {}
        eval_desc = "Local Test Loss {:.8f} | Dice {:.2f} | HD {:.2f}"
        for idx, item in enumerate(self.test_loaders):
            metric = Accumulator(5)
            eval_bar = tqdm.tqdm(initial=0, leave=True, total=len(item),
                                 desc=eval_desc.format(0, 0, 0), position=0)
            for data, label, mask in item:
                data, label = data.to(self._device), label.to(self._device)
                with torch.no_grad():
                    pred_score = self._model(data)
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
                    pred_score = self._model(data)
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

    def evaluate(self, evaluator):
        pass


class FedAvgSerialClientTrainer(SGDSerialClientTrainer):
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
        super(FedAvgSerialClientTrainer, self).__init__(model, num_clients, train_loaders, test_loaders, device, logger, personal)
        self._model = deepcopy(model).to(self._device)
        self.lr = lr
        self.criterion = criterion
        self.max_epoch = max_epoch
        self.optimizer = torch.optim.SGD(self._model.parameters(), self.lr)
        self.output_path = output_path
        self.current_round = 0
        self.evaluators = evaluators
        self.dice_macro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        self.dice_micro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        for idx in range(1, self.num_clients + 1):
            guarantee_path(self.output_path + "client" + str(idx) + "/train/")
            guarantee_path(self.output_path + "client" + str(idx) + "/local_test/")
            guarantee_path(self.output_path + "client" + str(idx) + "/global_test/")

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

    def local_test(self, client_idx, epoch: int):
        self._model.eval()
        l_metric_dict = {}
        eval_desc = "Local Test Loss {:.8f} | Dice {:.2f} | HD {:.2f}"
        for idx, item in enumerate(self.test_loaders):
            metric = Accumulator(5)
            eval_bar = tqdm.tqdm(initial=0, leave=True, total=len(item),
                                 desc=eval_desc.format(0, 0, 0), position=0)
            for data, label, mask in item:
                data, label = data.to(self._device), label.to(self._device)
                with torch.no_grad():
                    pred_score = self._model(data)
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
                    f"client{client_idx + 1}_client{idx + 1}_local_test_loss": metric[0] / metric[-1],
                    f"client{client_idx + 1}_client{idx + 1}_test_micro_dice": metric[1] / metric[-1],
                    f"client{client_idx + 1}_client{idx + 1}_test_macro_dice": metric[2] / metric[-1],
                    f"client{client_idx + 1}_client{idx + 1}_test_hd": metric[3] / metric[-1]
                }
            )
            l_metric_dict[str(idx)] = metric_dict
            self._LOGGER[client_idx].info(f"Epoch {epoch} | Client {idx + 1} Local Test Loss: {metric[0] / metric[-1]} | Local Test Dice: {metric[2] / metric[-1]}")
        self.evaluators[client_idx].add_dict("local_test", self.current_round, epoch, l_metric_dict)

    def global_test(self, idx, epoch: int):
        self._model.eval()
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
                    pred_score = self._model(data)
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
                f"client{idx + 1}_global_test_loss": metric[0] / metric[-1],
                f"client{idx + 1}_global_test_micro_dice": metric[1] / metric[-1],
                f"client{idx + 1}_global_test_macro_dice": metric[2] / metric[-1],
                f"client{idx + 1}_global_test_hd": metric[3] / metric[-1]
            }
        )
        self.evaluators[idx].add_dict("global_test", self.current_round, epoch, metric_dict)
        self._LOGGER[idx].info(f"Epoch {epoch} | Global Test Loss: {metric[0] / metric[-1]} | Global Test Dice: {metric[2] / metric[-1]}")
