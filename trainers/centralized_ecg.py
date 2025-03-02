
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.ecg.centralized import CentralizedSGDTrainer
from fedlab.utils.functional import setup_seed
from fedlab.utils.logger import Logger
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn as nn
from model import get_model
from utils.evaluation import MultiLabelEvaluator
from utils.dataloader import get_ecg_dataset
from utils.io import guarantee_path
import json
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epoch", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--model", type=str, default="resnet1d34")
parser.add_argument("--mode", type=str, default="centralized")
parser.add_argument("--case_name", type=str, default="centralized_ecg")
parser.add_argument("--optimizer_name", type=str, default="SGD")
parser.add_argument("--clients", type=list[str], default=["client1", "client2", "client3", "client4"])

if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)

    max_epoch = args.max_epoch
    batch_size = args.batch_size
    lr = args.lr
    input_path = args.input_path
    output_path = args.output_path
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    output_path = output_path if output_path[-1] == "/" else output_path + "/"
    output_path = output_path + args.model + "/" + args.mode + "/" + datetime.now().strftime("%Y%m%d%H%M%S") + "/"
    clients = args.clients

    guarantee_path(output_path)

    train_dataset = get_ecg_dataset(
        [
            f"{input_path}/ECG/preprocessed/{client}/train.csv" for client in clients
        ],
        base_path=f"{input_path}/ECG/preprocessed/",
        locations=clients,
        file_name="records.h5",
        n_classes=20
    )
    test_datasets = [get_ecg_dataset(
        [f"{input_path}/ECG/preprocessed/{client}/test.csv"],
        base_path=f"{input_path}/ECG/preprocessed/",
        locations=[client],
        file_name="records.h5",
        n_classes=20
    ) for client in clients]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loaders = [
        DataLoader(test_dataset, batch_size=batch_size,shuffle=False) for test_dataset in test_datasets
    ]
    model = get_model(args.model)
    criterion = nn.BCELoss()
    evaluator = MultiLabelEvaluator()

    setting = {
        "dataset": "ECG",
        "model": args.model,
        "batch_size": batch_size,
        "lr": lr,
        "criterion": "BCELoss",
        "seed": args.seed,
        "max_epoch": max_epoch
    }
    with open(output_path + "setting.json", "w") as f:
        f.write(json.dumps(setting))

    wandb.init(
        project="FedCVD_ECG_FL",
        name=args.case_name,
        config=setting
    )

    logger = Logger(log_name="centralized", log_file=output_path + "logger.log")

    trainer = CentralizedSGDTrainer(
        model=model,
        train_loader=train_loader,
        test_loaders=test_loaders,
        lr=lr,
        criterion=criterion,
        evaluator=evaluator,
        max_epoch=max_epoch,
        output_path=output_path,
        optimizer_name=args.optimizer_name,
        device=None,
        logger=logger
    )
    trainer.run(evaluator)
