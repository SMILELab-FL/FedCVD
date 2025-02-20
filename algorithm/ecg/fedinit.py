
from copy import deepcopy
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

    def local_process(self, payload, id_list):
        pack = None
        for idx in id_list:
            global_model = payload[0]
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
        wandb.log(
            {
                f"client_client{idx + 1}_train_loss": metric[0] / metric[2],
                f"client_client{idx + 1}_train_acc": metric[1] / metric[2],
                f"client_client{idx + 1}_train_micro_f1": metric_dict["micro_f1"],
                f"client_client{idx + 1}_train_mAP": float(np.average(metric_dict["average_precision_score"]))
            },
            step=self.current_round
        )
        return [self.model_parameters]
