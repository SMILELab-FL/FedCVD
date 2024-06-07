
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

import torch.nn as nn
from fedlab.utils.functional import setup_seed
from fedlab.utils.logger import Logger
from torch.utils.data import DataLoader

from algorithm.ecg.fedprox import FedProxServerHandler, FedProxSerialClientTrainer
from algorithm.pipeline import Pipeline
from model.resnet import resnet1d34
from utils.dataloader import get_ecg_dataset
from utils.evaluation import FedClientMultiLabelEvaluator, FedServerMultiLabelEvaluator
from utils.io import guarantee_path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epoch", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--mu", type=float, default=0.01)
parser.add_argument("--model", type=str, default="resnet1d34")
parser.add_argument("--mode", type=str, default="fedprox")
parser.add_argument("--communication_round", type=int, default=50)
parser.add_argument("--num_clients", type=int, default=4)
parser.add_argument("--clients", type=list[str], default=["client1", "client2", "client3", "client4"])

if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)

    batch_size = args.batch_size
    lr = args.lr
    mu = args.mu
    max_epoch = args.max_epoch
    communication_round = args.communication_round
    num_clients = args.num_clients
    sample_ratio = 1

    input_path = args.input_path
    output_path = args.output_path
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    output_path = output_path if output_path[-1] == "/" else output_path + "/"
    output_path = output_path + args.mode + "/" + datetime.now().strftime("%Y%m%d%H%M%S") + "/"
    clients = args.clients

    train_datasets = [get_ecg_dataset(
        [
            f"{input_path}/ECG/preprocessed/{client}/train.csv"
        ],
        base_path=f"{input_path}/ECG/preprocessed/",
        locations=clients,
        file_name="records.h5",
        n_classes=20
    ) for client in clients]
    test_datasets = [get_ecg_dataset(
        [f"{input_path}/ECG/preprocessed/{client}/test.csv"],
        base_path=f"{input_path}/ECG/preprocessed/",
        locations=clients,
        file_name="records.h5",
        n_classes=20
    ) for client in clients]

    train_loaders = [
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True) for train_dataset in train_datasets
    ]
    test_loaders = [
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False) for test_dataset in test_datasets
    ]
    model = resnet1d34()
    criterion = nn.BCELoss()
    client_evaluators = [FedClientMultiLabelEvaluator() for _ in range(1, 5)]
    server_evaluator = FedServerMultiLabelEvaluator()
    for client in clients:
        guarantee_path(output_path + client + "/")
    guarantee_path(output_path + "server/")

    setting = {
        "dataset": "ECG",
        "model": "resnet1d34",
        "batch_size": batch_size,
        "client_lr": lr,
        "mu": mu,
        "criterion": "BCELoss",
        "num_clients": num_clients,
        "sample_ratio": sample_ratio,
        "communication_round": communication_round,
        "max_epoch": max_epoch,
        "seed": args.seed
    }
    with open(output_path + "setting.json", "w") as f:
        f.write(json.dumps(setting))

    client_loggers = [
        Logger(log_name=client, log_file=output_path + client + "/logger.log") for client in clients
    ]
    server_logger = Logger(log_name="server", log_file=output_path + "server/logger.log")

    trainer = FedProxSerialClientTrainer(
        mu=mu,
        model=model,
        num_clients=num_clients,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        lr=lr,
        criterion=criterion,
        max_epoch=max_epoch,
        output_path=output_path,
        evaluators=client_evaluators,
        device=None,
        logger=client_loggers
    )

    handler = FedProxServerHandler(
        model=model,
        test_loaders=test_loaders,
        criterion=criterion,
        output_path=output_path,
        evaluator=server_evaluator,
        communication_round=communication_round,
        num_clients=num_clients,
        sample_ratio=1,
        device=None,
        logger=server_logger
    )
    standalone = Pipeline(handler, trainer)
    standalone.main()
