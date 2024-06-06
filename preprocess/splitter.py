
import random
import tqdm
# import hydra
import numpy as np
import pandas as pd
import argparse

def split(
        input_path: str,
        sample_ratio: float = 0.8,
        seed: int = 2
):
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    meta = pd.read_csv(input_path + "metadata.csv")
    train_data = meta.sample(frac=sample_ratio, random_state=seed)
    test_data = meta.drop(labels=train_data.index)
    train_data.to_csv(input_path + "train.csv", index=False, encoding="utf-8")
    test_data.to_csv(input_path + "test.csv", index=False, encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="data/")
parser.add_argument("--sample_ratio", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=2)

if __name__ == "__main__":
    args = parser.parse_args()
    split(args.input_path, args.sample_ratio, args.seed)
