
from copy import deepcopy
from typing import List

from algorithm.ecg.fedavg import FedAvgServerHandler, FedAvgSerialClientTrainer
from utils.evaluation import calculate_accuracy, calculate_multilabel_metrics, get_pred_label, transfer_tensor_to_numpy
from utils.evaluation import Accumulator
from fedlab.utils.logger import Logger
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators

import torch
import tqdm
import pandas as pd
import numpy as np
import wandb


class FedSMServerHandler(FedAvgServerHandler):
    def __init__(
        self,
        lambda_,
        gamma,
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
        super(FedSMServerHandler, self).__init__(model, test_loaders, criterion, output_path, evaluator, communication_round, num_clients, sample_ratio, device, logger)
        self.lambda_ = lambda_
        self.gamma = gamma
        self.local_models = [deepcopy(model).to(self._device) for _ in range(self.num_clients)]
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
            "local_models": [ele.state_dict() for ele in self.local_models],
            "model_selector": self.model_selector.state_dict(),
            "model": self._model.state_dict()
        }, path)

    def finish(self):
        self.evaluator.save(self.output_path + "server/metric.json")
        self.save_model(self.output_path + "server/model.pth")

    def local_test(self):
        self._model.eval()
        self.model_selector.eval()
        for idx in range(self.num_clients):
            self.local_models[idx].eval()
        l_metric_dict = {}
        eval_desc = "Local Test Loss {:.8f}  |  Acc:{:.2f}"
        for idx, item in enumerate(self.test_loaders):
            metric = Accumulator(3)
            eval_bar = tqdm.tqdm(initial=0, leave=True, total=len(item),
                                 desc=eval_desc.format(0, 0), position=0)
            pred_score_list = []
            pred_label_list = []
            true_label_list = []
            for data, label in item:
                data, label = data.to(self._device), label.to(self._device)
                with torch.no_grad():
                    select_score = self.model_selector(data)
                    max_select_score, argmax_select_score = torch.max(select_score, dim=1)
                    argmax_select_score = torch.where(
                        max_select_score > self.gamma, argmax_select_score, -1
                    )
                    global_batch = torch.eq(argmax_select_score, -1).flatten()
                    local_batch = [torch.eq(argmax_select_score, i).flatten() for i in range(self.num_clients)]
                    pred_score = torch.zeros(label.size(0), label.size(1)).to(self._device)

                    pred_score[global_batch] = self._model(data[global_batch])
                    for i in range(self.num_clients):
                        if torch.sum(local_batch[i]) > 0:
                            pred_score[local_batch[i]] = self.local_models[i](data[local_batch[i]])

                    pred_score_np = transfer_tensor_to_numpy(pred_score)
                    pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                    true_label_np = transfer_tensor_to_numpy(label)

                    pred_score_list.append(pred_score_np)
                    pred_label_list.append(pred_label_np)
                    true_label_list.append(true_label_np)

                    loss = self.criterion(pred_score, label)

                    metric.add(
                        float(loss) * len(label), calculate_accuracy(pred_label_np, true_label_np), len(label)
                    )

                eval_bar.desc = eval_desc.format(metric[0] / metric[2], metric[1] / metric[2])
                eval_bar.update(1)
            eval_bar.close()
            all_pred_score_np = np.concatenate(pred_score_list, axis=0)
            all_pred_label_np = np.concatenate(pred_label_list, axis=0)
            all_true_label_np = np.concatenate(true_label_list, axis=0)
            df = pd.DataFrame(all_pred_score_np)
            df.to_csv(
                self.output_path + "server/" + "local_test/local_pred_score_" + str(idx) + ".csv", index=False, encoding="utf-8"
            )
            df = pd.DataFrame(all_pred_label_np)
            df.to_csv(
                self.output_path + "server/" + "local_test/local_pred_label_" + str(idx) + ".csv", index=False, encoding="utf-8"
            )
            df = pd.DataFrame(all_true_label_np)
            df.to_csv(
                self.output_path + "server/" + "local_test/local_true_label_" + str(idx) + ".csv", index=False, encoding="utf-8"
            )
            metric_dict = calculate_multilabel_metrics(all_pred_score_np, all_pred_label_np, all_true_label_np)
            metric_dict["loss"] = metric[0] / metric[2]
            l_metric_dict[str(idx)] = metric_dict
            self._LOGGER.info(f"Client {idx + 1} Local Test Loss: {metric[0] / metric[2]} | Local Test Acc: {metric[1] / metric[2]}")
            wandb.log(
                {
                    f"server_client{idx + 1}_local_test_loss": metric[0] / metric[2],
                    f"server_client{idx + 1}_local_test_acc": metric[1] / metric[2],
                    f"server_client{idx + 1}_local_test_micro_f1": metric_dict["micro_f1"],
                    f"server_client{idx + 1}_local_test_mAP": float(np.average(metric_dict["average_precision_score"]))
                },
                step=self.round
            )
        self.evaluator.add_dict("local_test", self.round, l_metric_dict)

    def global_test(self):
        self._model.eval()
        self.model_selector.eval()
        for idx in range(self.num_clients):
            self.local_models[idx].eval()
        metric = Accumulator(3)
        pred_score_list = []
        pred_label_list = []
        true_label_list = []
        eval_desc = " Global Test Loss {:.8f}  |  Acc:{:.2f}"
        length = 0
        for item in self.test_loaders:
            length += len(item)
        eval_bar = tqdm.tqdm(initial=0, leave=True, total=length,
                        desc=eval_desc.format(0, 0), position=0)
        for item in self.test_loaders:
            for data, label in item:
                data, label = data.to(self._device), label.to(self._device)
                with torch.no_grad():
                    select_score = self.model_selector(data)
                    max_select_score, argmax_select_score = torch.max(select_score, dim=1)
                    argmax_select_score = torch.where(
                        max_select_score > self.gamma, argmax_select_score, -1
                    )
                    global_batch = torch.eq(argmax_select_score, -1).flatten()
                    local_batch = [torch.eq(argmax_select_score, i).flatten() for i in range(self.num_clients)]
                    pred_score = torch.zeros(label.size(0), label.size(1)).to(self._device)

                    pred_score[global_batch] = self._model(data[global_batch])
                    for i in range(self.num_clients):
                        if torch.sum(local_batch[i]) > 0:
                            pred_score[local_batch[i]] = self.local_models[i](data[local_batch[i]])

                    pred_score_np = transfer_tensor_to_numpy(pred_score)
                    pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                    true_label_np = transfer_tensor_to_numpy(label)

                    pred_score_list.append(pred_score_np)
                    pred_label_list.append(pred_label_np)
                    true_label_list.append(true_label_np)

                    loss = self.criterion(pred_score, label)

                    metric.add(
                        float(loss) * len(label), calculate_accuracy(pred_label_np, true_label_np), len(label)
                    )

                eval_bar.desc = eval_desc.format(metric[0] / metric[2], metric[1] / metric[2])
                eval_bar.update(1)
        eval_bar.close()
        all_pred_score_np = np.concatenate(pred_score_list, axis=0)
        all_pred_label_np = np.concatenate(pred_label_list, axis=0)
        all_true_label_np = np.concatenate(true_label_list, axis=0)
        df = pd.DataFrame(all_pred_score_np)
        df.to_csv(
            self.output_path + "server/" + "global_test/local_pred_score.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_pred_label_np)
        df.to_csv(
            self.output_path + "server/" + "global_test/local_pred_label.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_true_label_np)
        df.to_csv(
            self.output_path + "server/" + "global_test/local_true_label.csv", index=False, encoding="utf-8"
        )
        metric_dict = calculate_multilabel_metrics(all_pred_score_np, all_pred_label_np, all_true_label_np)
        metric_dict["loss"] = metric[0] / metric[2]
        self.evaluator.add_dict("global_test", self.round, metric_dict)
        self._LOGGER.info(f"Global Test Loss: {metric[0] / metric[2]} | Global Test Acc: {metric[1] / metric[2]}")
        wandb.log(
            {
                f"server_global_test_loss": metric[0] / metric[2],
                f"server_global_test_acc": metric[1] / metric[2],
                f"server_global_test_micro_f1": metric_dict["micro_f1"],
                f"server_global_test_mAP": float(np.average(metric_dict["average_precision_score"]))
            },
            step=self.round
        )


class FedSMSerialClientTrainer(FedAvgSerialClientTrainer):
    def __init__(
            self,
            model,
            model_selector,
            num_clients,
            train_loaders,
            test_loaders,
            lr: float,
            criterion: torch.nn.Module,
            max_epoch: int,
            output_path: str,
            evaluators,
            device: torch.device | None = None,
            logger=None,
            personal=False
    ):
        super(FedSMSerialClientTrainer, self).__init__(
            model=model,
            num_clients=num_clients,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            lr=lr,
            criterion=criterion,
            max_epoch=max_epoch,
            output_path=output_path,
            evaluators=evaluators,
            device=device,
            logger=logger,
            personal=personal
        )
        self.model_selector = deepcopy(model_selector).to(self._device)
        self.local_model = deepcopy(model).to(self._device)
        self.global_models = []
        self.local_models = [self.model_parameters for _ in range(self.num_clients)]
        self.s_criterion = torch.nn.CrossEntropyLoss()
        self.s_optimizer = torch.optim.SGD(self.model_selector.parameters(), lr=lr)

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

        metric = Accumulator(2)
        train_desc = "Epoch {:2d}: train Loss {:.8f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loaders[idx]),
                              desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label in self.train_loaders[idx]:
            loc = torch.tensor([idx for _ in range(len(data))], dtype=torch.long)
            data, label, loc = data.to(self._device), label.to(self._device), loc.to(self._device)

            g_loss = self.criterion(self._model(data), label)
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

            metric.add(
                float(g_loss) * len(label), len(label)
            )
            train_bar.desc = train_desc.format(epoch, metric[0] / metric[-1])
            train_bar.update(1)
        train_bar.close()
        self._LOGGER[idx].info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[-1]}")

        return [
            self.model_parameters,
            SerializationTool.serialize_model(self.local_model),
            SerializationTool.serialize_model(self.model_selector),
            metric[-1]
        ]

    def train_global_model(self, epoch, idx):
        self._model.train()

        metric = Accumulator(3)
        train_desc = "Epoch {:2d}: train Loss {:.8f}  |  Acc:{:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loaders[idx]),
                              desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label in self.train_loaders[idx]:
            data, label = data.to(self._device), label.to(self._device)

            pred_score = self._model(data)
            with torch.no_grad():
                pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                true_label_np = transfer_tensor_to_numpy(label)

            loss = self.criterion(pred_score, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metric.add(
                float(loss) * len(label), calculate_accuracy(pred_label_np, true_label_np), len(label)
            )
            train_bar.desc = train_desc.format(epoch, metric[0] / metric[2], metric[1] / metric[2])
            train_bar.update(1)
        train_bar.close()
        self._LOGGER[idx].info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[2]} | Train Acc: {metric[1] / metric[2]}")

        return [self.model_parameters, metric[2]]

    def train_local_model(self, epoch, idx):
        self._model.train()

        metric = Accumulator(3)
        pred_score_list = []
        pred_label_list = []
        true_label_list = []
        train_desc = "Epoch {:2d}: train Loss {:.8f}  |  Acc:{:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loaders[idx]),
                              desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label in self.train_loaders[idx]:
            data, label = data.to(self._device), label.to(self._device)

            pred_score = self._model(data)
            with torch.no_grad():
                pred_score_np = transfer_tensor_to_numpy(pred_score)
                pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                true_label_np = transfer_tensor_to_numpy(label)
                pred_score_list.append(pred_score_np)
                pred_label_list.append(pred_label_np)
                true_label_list.append(true_label_np)

            loss = self.criterion(pred_score, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metric.add(
                float(loss) * len(label), calculate_accuracy(pred_label_np, true_label_np), len(label)
            )
            train_bar.desc = train_desc.format(epoch, metric[0] / metric[2], metric[1] / metric[2])
            train_bar.update(1)
        train_bar.close()
        all_pred_score_np = np.concatenate(pred_score_list, axis=0)
        all_pred_label_np = np.concatenate(pred_label_list, axis=0)
        all_true_label_np = np.concatenate(true_label_list, axis=0)
        df = pd.DataFrame(all_pred_score_np)
        df.to_csv(
            self.output_path + "client" + str(idx + 1) + "/train/local_pred_score.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_pred_label_np)
        df.to_csv(
            self.output_path + "client" + str(idx + 1) + "/train/local_pred_label.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_true_label_np)
        df.to_csv(
            self.output_path + "client" + str(idx + 1) + "/train/local_true_label.csv", index=False, encoding="utf-8"
        )
        metric_dict = calculate_multilabel_metrics(all_pred_score_np, all_pred_label_np, all_true_label_np)
        metric_dict["loss"] = metric[0] / metric[2]
        self.evaluators[idx].add_dict("train", self.current_round, epoch, metric_dict)
        self._LOGGER[idx].info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[2]} | Train Acc: {metric[1] / metric[2]}")

        return [self.model_parameters, metric[2]]

    def train_selector(self, epoch, idx):
        self.model_selector.train()

        metric = Accumulator(2)
        train_desc = "Epoch {:2d}: train Loss {:.8f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loaders[idx]),
                              desc=train_desc.format(epoch, 0, 0), position=0)
        for data, _ in self.train_loaders[idx]:
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
            train_bar.desc = train_desc.format(epoch, metric[0] / metric[1])
            train_bar.update(1)
        train_bar.close()
        self._LOGGER[idx].info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[1]}")

        return [SerializationTool.serialize_model(self.model_selector), metric[1]]
