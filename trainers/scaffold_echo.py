
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.echo.scaffold import ScaffoldSerialClientTrainer, ScaffoldServerHandler
from algorithm.pipeline import Pipeline
from fedlab.utils.functional import setup_seed
from fedlab.utils.logger import Logger
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn as nn
from model import get_model
from utils.evaluation import FedClientMultiLabelEvaluator, FedServerMultiLabelEvaluator
from utils.dataloader import get_echo_dataset
from utils.io import guarantee_path
import json
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--server_lr", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--communication_round", type=int, default=50)
parser.add_argument("--max_epoch", type=int, default=1)
parser.add_argument("--n_classes", type=int, default=4)
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--case_name", type=str, default="scaffold_echo")
parser.add_argument("--num_clients", type=int, default=3)
parser.add_argument("--mode", type=str, default="scaffold")
parser.add_argument("--optimizer_name", type=str, default="SGD")
parser.add_argument("--clients", type=list[str], default=["client1", "client2", "client3"])
parser.add_argument("--frac", type=float, default=1.0)

if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)

    max_epoch = args.max_epoch
    communication_round = args.communication_round
    batch_size = args.batch_size
    lr = args.lr
    server_lr = args.server_lr
    num_clients = args.num_clients
    sample_ratio = 1

    input_path = args.input_path
    output_path = args.output_path
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    output_path = output_path if output_path[-1] == "/" else output_path + "/"
    output_path = output_path + args.model + "/" + args.mode + "/" + datetime.now().strftime("%Y%m%d%H%M%S") + "/"
    clients = args.clients

    guarantee_path(output_path)

    train_datasets = [get_echo_dataset(
        [
            f"{input_path}/ECHO/preprocessed/{client}/train.csv"
        ],
        base_path=f"{input_path}/ECHO/preprocessed/",
        locations=clients,
        file_name="records.h5",
        n_classes=args.n_classes
    ) for client in clients]
    test_datasets = [get_echo_dataset(
        [f"{input_path}/ECHO/preprocessed/{client}/test.csv"],
        base_path=f"{input_path}/ECHO/preprocessed/",
        locations=clients,
        file_name="records.h5",
        n_classes=args.n_classes
    ) for client in clients]

    train_loaders = [
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True) for train_dataset in train_datasets
    ]
    test_loaders = [
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False) for test_dataset in test_datasets
    ]
    model = get_model(args.model, n_classes=args.n_classes)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    client_evaluators = [FedClientMultiLabelEvaluator() for _ in range(1, 4)]
    server_evaluator = FedServerMultiLabelEvaluator()

    for client in clients:
        guarantee_path(output_path + client + "/")
    guarantee_path(output_path + "server/")

    setting = {
        "dataset": "ECHO",
        "model": args.model,
        "batch_size": batch_size,
        "client_lr": lr,
        "server_lr": server_lr,
        "criterion": "CELoss",
        "num_clients": num_clients,
        "sample_ratio": sample_ratio,
        "communication_round": communication_round,
        "max_epoch": max_epoch,
        "seed": args.seed
    }
    with open(output_path + "setting.json", "w") as f:
        f.write(json.dumps(setting))

    wandb.init(
        project="FedCVD_ECHO_FL",
        name=args.case_name,
        config={
            "dataset": "ECHO",
            "model": args.model,
            "batch_size": batch_size,
            "lr": lr,
            "criterion": "CELoss",
            "max_epoch": max_epoch,
            "seed": args.seed
        }
    )

    client_loggers = [
        Logger(log_name=client, log_file=output_path + client + "/logger.log") for client in clients
    ]
    server_logger = Logger(log_name="server", log_file=output_path + "server/logger.log")

    trainer = ScaffoldSerialClientTrainer(
        model=model,
        num_clients=num_clients,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_classes=args.n_classes,
        lr=lr,
        criterion=criterion,
        max_epoch=max_epoch,
        output_path=output_path,
        evaluators=client_evaluators,
        optimizer_name=args.optimizer_name,
        device=None,
        logger=client_loggers
    )

    handler = ScaffoldServerHandler(
        lr=server_lr,
        num_classes=args.n_classes,
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
