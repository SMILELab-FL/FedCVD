
from typing import List

from copy import deepcopy
from torch.nn import Module
from torch.utils.data import DataLoader
from fedlab.utils.functional import get_best_device
from fedlab.utils.logger import Logger
from utils.evaluation import Accumulator
from utils.evaluation import transfer_tensor_to_numpy, calculate_accuracy, get_pred_label, calculate_multilabel_metrics
from utils.evaluation import shield, cal_hd, generate_pseudo_label
from utils.io import guarantee_path
from torchmetrics import Dice

import torch
import numpy as np
import tqdm
import pandas as pd
import wandb

class CentralizedSGDTrainer:
    def __init__(
            self,
            model: Module,
            train_loader: DataLoader,
            test_loaders: List[DataLoader],
            lr: float,
            criterion: Module,
            evaluator,
            max_epoch: int,
            output_path: str,
            num_classes: int,
            device: torch.device | None = None,
            logger: Logger | None = None
    ):
        self.train_loader = train_loader
        self.test_loaders = test_loaders
        self.lr = lr
        self.criterion = criterion
        self.max_epoch = max_epoch
        self._device = get_best_device() if device is None else device
        self._model = deepcopy(model).to(self._device)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self._LOGGER = Logger() if logger is None else logger
        self.output_path = output_path
        self.evaluator = evaluator
        self.dice_macro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        self.dice_micro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        guarantee_path(self.output_path + "train/")
        guarantee_path(self.output_path + "local_test/")
        guarantee_path(self.output_path + "global_test/")

    @property
    def model(self):
        return self._model

    def run(self, evaluator):
        for epoch in range(self.max_epoch):
            self.train(epoch, evaluator)
            self.local_test(evaluator, epoch)
            self.global_test(evaluator, epoch)


    def finish(self):
        self.evaluator.save(self.output_path + "metric.json")
        torch.save({
            "model": self._model.state_dict()
        },
            self.output_path + "model.pth"
        )
        self._LOGGER.close()

    def train(self, epoch, evaluator):
        self._model.train()
        metric = Accumulator(5)
        train_desc = "Epoch {:2d}: train Loss {:.8f} | Dice {:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loader),
                         desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label, mask in self.train_loader:
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
                "train_loss": metric[0] / metric[-1],
                "train_micro_dice": metric[1] / metric[-1],
                "train_macro_dice": metric[2] / metric[-1],
                "train_hd": metric[3] / metric[-1]
            }
        )
        evaluator.add_dict("train", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[-1]} | Train Dice: {metric[2] / metric[-1]}")

    def local_test(self, evaluator, epoch: int):
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
            self._LOGGER.info(f"Epoch {epoch} | Client {idx + 1} Local Test Loss: {metric[0] / metric[-1]} | Local Test Dice: {metric[2] / metric[-1]}")
        evaluator.add_dict("local_test", epoch, l_metric_dict)

    def global_test(self, evaluator, epoch: int):
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
        evaluator.add_dict("global_test", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Global Test Loss: {metric[0] / metric[-1]} | Global Test Dice: {metric[2] / metric[-1]}")


class CentralizedIgnoreSGDTrainer:
    def __init__(
            self,
            model: Module,
            train_loader: DataLoader,
            test_loaders: List[DataLoader],
            lr: float,
            criterion: Module,
            evaluator,
            max_epoch: int,
            output_path: str,
            num_classes: int,
            device: torch.device | None = None,
            logger: Logger | None = None
    ):
        self.train_loader = train_loader
        self.test_loaders = test_loaders
        self.lr = lr
        self.criterion = criterion
        self.max_epoch = max_epoch
        self._device = get_best_device() if device is None else device
        self._model = deepcopy(model).to(self._device)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self._LOGGER = Logger() if logger is None else logger
        self.output_path = output_path
        self.evaluator = evaluator
        self.dice_macro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        self.dice_micro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        guarantee_path(self.output_path + "train/")
        guarantee_path(self.output_path + "local_test/")
        guarantee_path(self.output_path + "global_test/")
    @property
    def model(self):
        return self._model

    def run(self, evaluator):
        for epoch in range(self.max_epoch):
            self.train(epoch, evaluator)
            self.local_test(evaluator, epoch)
            self.global_test(evaluator, epoch)
        self.finish()


    def finish(self):
        self.evaluator.save(self.output_path + "metric.json")
        torch.save({
            "model": self._model.state_dict()
        },
            self.output_path + "model.pth"
        )
        self._LOGGER.close()

    def train(self, epoch,evaluator):
        self._model.train()
        metric = Accumulator(5)
        train_desc = "Epoch {:2d}: train Loss {:.8f} | Dice {:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loader),
                              desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label, mask in self.train_loader:
            # train_label = deepcopy(label)
            unlabeled_batch = torch.ne(mask, 0).flatten()
            # train_label[unlabeled_batch] = torch.where(train_label[unlabeled_batch] == 0, -1, train_label[unlabeled_batch])

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
        #         "train_loss": metric[0] / metric[-1],
        #         "train_micro_dice": metric[1] / metric[-1],
        #         "train_macro_dice": metric[2] / metric[-1],
        #         "train_hd": metric[3] / metric[-1]
        #     }
        # )
        # evaluator.add_dict("train", epoch, metric_dict)
        # self._LOGGER.info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[-1]} | Train Dice: {metric[2] / metric[-1]}")

    def local_test(self, evaluator, epoch: int):
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
            self._LOGGER.info(f"Epoch {epoch} | Client {idx + 1} Local Test Loss: {metric[0] / metric[-1]} | Local Test Dice: {metric[2] / metric[-1]}")
        evaluator.add_dict("local_test", epoch, l_metric_dict)

    def global_test(self, evaluator, epoch: int):
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
        evaluator.add_dict("global_test", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Global Test Loss: {metric[0] / metric[-1]} | Global Test Dice: {metric[2] / metric[-1]}")


class CentralizedSemiSGDTrainer:
    def __init__(
            self,
            model: Module,
            labeled_loader: DataLoader,
            train_loader: DataLoader,
            test_loaders: List[DataLoader],
            lr: float,
            criterion: Module,
            evaluator,
            max_epoch: int,
            output_path: str,
            num_classes: int,
            device: torch.device | None = None,
            logger: Logger | None = None
    ):
        self.labeled_loader = labeled_loader
        self.train_loader = train_loader
        self.test_loaders = test_loaders
        self.lr = lr
        self.criterion = criterion
        self.max_epoch = max_epoch
        self._device = get_best_device() if device is None else device
        self._model = deepcopy(model).to(self._device)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self._LOGGER = Logger() if logger is None else logger
        self.output_path = output_path
        self.evaluator = evaluator
        self.dice_macro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        self.dice_micro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        guarantee_path(self.output_path + "train/")
        guarantee_path(self.output_path + "local_test/")
        guarantee_path(self.output_path + "global_test/")

    def alpha(self, t, t1=10, t2=40, weight=1):
        if t < t1:
            return 0
        elif t < t2:
            return (t - t1) / (t2 - t1) * weight
        return weight
    @property
    def model(self):
        return self._model

    def run(self, evaluator, t1=10, t2=40, weight=1):
        for epoch in range(self.max_epoch):
            self.train(epoch, evaluator, t1=t1, t2=t2, weight=weight)
            self.local_test(evaluator, epoch)
            self.global_test(evaluator, epoch)
        self.finish()


    def finish(self):
        self.evaluator.save(self.output_path + "metric.json")
        torch.save({
            "model": self._model.state_dict()
        },
            self.output_path + "model.pth"
        )
        self._LOGGER.close()

    def train(self, epoch,evaluator, t1=10, t2=40, weight=1):
        self._model.train()
        metric = Accumulator(5)
        train_desc = "Epoch {:2d}: train Loss {:.8f} | Dice {:.2f}"
        if self.alpha(epoch, t1, t2, weight) == 0:
            train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.labeled_loader),
                             desc=train_desc.format(epoch, 0, 0), position=0)
            for data, label, mask in self.labeled_loader:
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
        else:
            train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loader),
                                  desc=train_desc.format(epoch, 0, 0), position=0)
            alp = self.alpha(epoch)
            for data, label, mask in self.train_loader:
                data, label = data.to(self._device), label.to(self._device)
                pred_score = self._model(data)
                labeled_batch = torch.eq(mask, 0).flatten()
                unlabeled_batch = torch.ne(mask, 0).flatten()
                if torch.sum(labeled_batch) == 0:
                    labeled_loss = 0
                else:
                    labeled_loss = self.criterion(pred_score[labeled_batch], label[labeled_batch])
                if torch.sum(unlabeled_batch) == 0:
                    unlabeled_loss = 0
                else:
                    with torch.no_grad():
                        pred_label = torch.argmax(pred_score, dim=1)
                        pseudo_label = generate_pseudo_label(pred_label, label, mask)
                    unlabeled_loss = self.criterion(pred_score[unlabeled_batch], pseudo_label[unlabeled_batch])
                loss = labeled_loss + alp * unlabeled_loss

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
                "train_loss": metric[0] / metric[-1],
                "train_micro_dice": metric[1] / metric[-1],
                "train_macro_dice": metric[2] / metric[-1],
                "train_hd": metric[3] / metric[-1]
            }
        )
        evaluator.add_dict("train", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[-1]} | Train Dice: {metric[2] / metric[-1]}")

    def local_test(self, evaluator, epoch: int):
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
            self._LOGGER.info(f"Epoch {epoch} | Client {idx + 1} Local Test Loss: {metric[0] / metric[-1]} | Local Test Dice: {metric[2] / metric[-1]}")
        evaluator.add_dict("local_test", epoch, l_metric_dict)

    def global_test(self, evaluator, epoch: int):
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
        evaluator.add_dict("global_test", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Global Test Loss: {metric[0] / metric[-1]} | Global Test Dice: {metric[2] / metric[-1]}")


class CentralizedSGDResNetTrainer:
    def __init__(
            self,
            model: Module,
            train_loader: DataLoader,
            test_loaders: List[DataLoader],
            lr: float,
            criterion: Module,
            evaluator,
            max_epoch: int,
            output_path: str,
            num_classes: int,
            device: torch.device | None = None,
            logger: Logger | None = None
    ):
        self.train_loader = train_loader
        self.test_loaders = test_loaders
        self.lr = lr
        self.criterion = criterion
        self.max_epoch = max_epoch
        self._device = get_best_device() if device is None else device
        self._model = deepcopy(model).to(self._device)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self._LOGGER = Logger() if logger is None else logger
        self.output_path = output_path
        self.evaluator = evaluator
        self.dice_macro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        self.dice_micro = Dice(ignore_index=0, num_classes=num_classes, average="macro").to(self._device)
        guarantee_path(self.output_path + "train/")
        guarantee_path(self.output_path + "local_test/")
        guarantee_path(self.output_path + "global_test/")

    @property
    def model(self):
        return self._model

    def run(self, evaluator):
        for epoch in range(self.max_epoch):
            self.train(epoch, evaluator)
            self.local_test(evaluator, epoch)
            self.global_test(evaluator, epoch)


    def finish(self):
        # self.evaluator.save(self.output_path + "metric.json")
        torch.save({
            "model": self._model.state_dict()
        },
            self.output_path + "model.pth"
        )
        self._LOGGER.close()

    def train(self, epoch, evaluator):
        self._model.train()
        metric = Accumulator(5)
        train_desc = "Epoch {:2d}: train Loss {:.8f} | Dice {:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loader),
                         desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label, mask in self.train_loader:
            data, label = data.to(self._device), label.to(self._device)

            pred_score = self._model(data)["out"]

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
                "train_loss": metric[0] / metric[-1],
                "train_micro_dice": metric[1] / metric[-1],
                "train_macro_dice": metric[2] / metric[-1],
                "train_hd": metric[3] / metric[-1]
            }
        )
        evaluator.add_dict("train", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[-1]} | Train Dice: {metric[2] / metric[-1]}")

    def local_test(self, evaluator, epoch: int):
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
                    pred_score = self._model(data)["out"]
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
            self._LOGGER.info(f"Epoch {epoch} | Client {idx + 1} Local Test Loss: {metric[0] / metric[-1]} | Local Test Dice: {metric[2] / metric[-1]}")
        evaluator.add_dict("local_test", epoch, l_metric_dict)

    def global_test(self, evaluator, epoch: int):
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
                    pred_score = self._model(data)["out"]
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
        evaluator.add_dict("global_test", epoch, metric_dict)
        self._LOGGER.info(f"Epoch {epoch} | Global Test Loss: {metric[0] / metric[-1]} | Global Test Dice: {metric[2] / metric[-1]}")