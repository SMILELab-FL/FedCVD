
from copy import deepcopy
from fedlab.utils import Aggregators
from fedlab.utils import SerializationTool
from fedlab.algorithm.base_client import SGDSerialClientTrainer
from fedlab.algorithm.base_server import SyncServerHandler
from utils.evaluation import Accumulator
from utils.evaluation import transfer_tensor_to_numpy, calculate_accuracy, get_pred_label, calculate_multilabel_metrics
from utils.io import guarantee_path
from fedlab.utils.logger import Logger

import torch
import tqdm
import numpy as np
import pandas as pd
import wandb


class FedAvgServerHandler(SyncServerHandler):
    """FedAvg server handler."""
    def __init__(
        self,
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
                    pred_score = self._model(data)
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
                    pred_score = self._model(data)
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

    def evaluate(self, evaluator):
        pass


class FedAvgSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(
            self,
            model,
            num_clients,
            train_loaders,
            test_loaders,
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
        super(FedAvgSerialClientTrainer, self).__init__(model, num_clients, train_loaders, test_loaders, device, logger, personal)
        self._model = deepcopy(model).to(self._device)
        self.lr = lr
        self.criterion = criterion
        self.max_epoch = max_epoch
        if optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(self._model.parameters(), self.lr)
        elif optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self._model.parameters(), self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        self.output_path = output_path
        self.current_round = 0
        self.evaluators = evaluators
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

    def local_test(self, client_idx, epoch):
        self._model.eval()
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
                    pred_score = self._model(data)
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
                self.output_path + "client" + str(client_idx + 1) + "/local_test/local_pred_score_" + str(idx) + ".csv", index=False, encoding="utf-8"
            )
            df = pd.DataFrame(all_pred_label_np)
            df.to_csv(
                self.output_path + "client" + str(client_idx + 1) + "/local_test/local_pred_label_" + str(idx) + ".csv", index=False, encoding="utf-8"
            )
            df = pd.DataFrame(all_true_label_np)
            df.to_csv(
                self.output_path + "client" + str(client_idx + 1) + "/local_test/local_true_label_" + str(idx) + ".csv", index=False, encoding="utf-8"
            )
            metric_dict = calculate_multilabel_metrics(all_pred_score_np, all_pred_label_np, all_true_label_np)
            metric_dict["loss"] = metric[0] / metric[2]
            l_metric_dict[str(idx)] = metric_dict
            self._LOGGER[client_idx].info(f"Epoch {epoch} | Client {idx + 1} Local Test Loss: {metric[0] / metric[2]} | Local Test Acc: {metric[1] / metric[2]}")
            wandb.log(
                {
                    f"client{client_idx + 1}_client{idx + 1}_local_test_loss": metric[0] / metric[2],
                    f"client{client_idx + 1}_client{idx + 1}_local_test_acc": metric[1] / metric[2],
                    f"client{client_idx + 1}_client{idx + 1}_local_test_micro_f1": metric_dict["micro_f1"],
                    f"client{client_idx + 1}_client{idx + 1}_local_test_mAP": float(np.average(metric_dict["average_precision_score"]))
                },
                step=self.current_round
            )
        self.evaluators[client_idx].add_dict("local_test", self.current_round, epoch, l_metric_dict)

    def global_test(self, idx, epoch):
        self._model.eval()
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
                    pred_score = self._model(data)
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
            self.output_path + "client" + str(idx + 1) + "/global_test/local_pred_score.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_pred_label_np)
        df.to_csv(
            self.output_path + "client" + str(idx + 1) + "/global_test/local_pred_label.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_true_label_np)
        df.to_csv(
            self.output_path + "client" + str(idx + 1) + "/global_test/local_true_label.csv", index=False, encoding="utf-8"
        )
        metric_dict = calculate_multilabel_metrics(all_pred_score_np, all_pred_label_np, all_true_label_np)
        metric_dict["loss"] = metric[0] / metric[2]
        self.evaluators[idx].add_dict("global_test", self.current_round, epoch, metric_dict)
        self._LOGGER[idx].info(f"Epoch {epoch} | Global Test Loss: {metric[0] / metric[2]} | Global Test Acc: {metric[1] / metric[2]}")
        wandb.log(
            {
                f"client{idx + 1}_global_test_loss": metric[0] / metric[2],
                f"client{idx + 1}_global_test_acc": metric[1] / metric[2],
                f"client{idx + 1}_global_test_micro_f1": metric_dict["micro_f1"],
                f"client{idx + 1}_global_test_mAP": float(np.average(metric_dict["average_precision_score"]))
            },
            step=self.current_round
        )
