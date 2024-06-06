
import os


def guarantee_path(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
