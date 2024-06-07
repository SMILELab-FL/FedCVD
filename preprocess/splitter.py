
import random
import tqdm
import os
import numpy as np
import pandas as pd
import argparse


def train_data_split(
        input_path: str,
        data_type: str,
        sample_ratio: float = 0.8,
        seed: int = 2
):
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    meta = pd.read_csv(input_path + "metadata.csv", dtype={data_type: str})
    train_data = meta.sample(frac=sample_ratio, random_state=seed)
    test_data = meta.drop(labels=train_data.index)
    train_data.to_csv(input_path + "train.csv", index=False, encoding="utf-8")
    test_data.to_csv(input_path + "test.csv", index=False, encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="data/")
parser.add_argument("--data_type", type=str, default="ECHO")
parser.add_argument("--sample_ratio", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=2)

if __name__ == "__main__":
    args = parser.parse_args()
    base_path = args.input_path
    base_path = base_path if base_path[-1] == "/" else base_path + "/"
    data_type = "ECG_ID" if args.data_type == "ECG" else "ECHO_ID"
    for client in os.listdir(base_path):
        if os.path.isdir(base_path + client):
            train_data_split(base_path + client, data_type, args.sample_ratio, args.seed)
