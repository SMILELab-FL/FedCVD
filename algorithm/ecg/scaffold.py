
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

class ScaffoldServerHandler(FedAvgServerHandler):
    def __init__(
        self,
        lr: float,
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

        return [self.model_parameters]
