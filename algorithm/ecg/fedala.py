
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
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
import random
import wandb


class FedALAServerHandler(FedAvgServerHandler):
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


class FedALASerialClientTrainer(FedAvgSerialClientTrainer):
    def __init__(
            self,
            rand_percent: int,
            model,
            num_clients,
            batch_size,
            train_datasets,
            train_loaders,
            test_loaders,
            lr: float,
            criterion: torch.nn.Module,
            max_epoch: int,
            output_path: str,
            evaluators,
            layer_idx: int = 0,
            eta: float = 1.0,
            threshold: float = 0.1,
            num_pre_loss: int = 10,
            optimizer_name: str = "SGD",
            device: torch.device | None = None,
            logger=None,
            personal=False
    ):
        super(FedALASerialClientTrainer, self).__init__(
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
        self.batch_size = batch_size
        self.train_datasets = train_datasets
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.global_models = []
        self.local_models = [self.model_parameters for _ in range(self.num_clients)]
        self.weights = [None for _ in range(self.num_clients)]
        self.start_phase = [True for _ in range(self.num_clients)]

    def local_process(self, payload, id_list):
        pack = None
        global_parameters = payload[0]
        for idx in id_list:
            local_parameters = self.adaptive_local_aggregation(global_parameters, self.local_models[idx], idx)
            self.set_model(local_parameters)
            for epoch in range(self.max_epoch):
                pack = self.train(epoch, idx)
                self.local_test(idx, epoch)
                self.global_test(idx, epoch)
            self.local_models[idx] = self.model_parameters
            self.cache.append(pack)
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
        return [self.model_parameters, metric[2]]

    def adaptive_local_aggregation(self,
                                   global_parameters,
                                   local_parameters,
                                   idx):
        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_datasets[idx]))
        rand_idx = random.randint(0, len(self.train_datasets[idx]) - rand_num)
        subset = Subset(self.train_datasets[idx], range(rand_idx, rand_idx + rand_num))
        rand_loader = DataLoader(
            subset, self.batch_size
        )

        # obtain the references of the parameters
        global_model = deepcopy(self._model)
        local_model = deepcopy(self._model)
        SerializationTool.deserialize_model(global_model, global_parameters)
        params_g = list(global_model.parameters())
        SerializationTool.deserialize_model(local_model, local_parameters)
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            assert torch.equal(SerializationTool.serialize_model(local_model), local_parameters)
            return SerializationTool.serialize_model(local_model)

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        model_t = deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights[idx] is None:
            self.weights[idx] = [torch.ones_like(param.data).to(self._device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                   self.weights[idx]):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            loss = 0
            num = 0
            for x, y in rand_loader:
                x, y = x.to(self._device), y.to(self._device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.criterion(output, y)  # modify according to the local objective
                loss_value.backward()
                with torch.no_grad():
                    loss += loss_value.item() * len(y)
                    num += len(y)

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                           params_gp, self.weights[idx]):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                           params_gp, self.weights[idx]):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss / num)
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase[idx]:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                self._LOGGER[idx].info(
                    f'Client:{idx + 1}\tStd: {np.std(losses[-self.num_pre_loss:])}\tALA epochs: {cnt}'
                )
                break

        self.start_phase[idx] = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
        return SerializationTool.serialize_model(local_model)
