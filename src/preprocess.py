"""Stage 2 - normalize to [0, 1], split a validation set, save to data/processed/."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def normalize(x):
    x = x.astype("float32") / 255.0
    return (x - 0.2860) / 0.3530      # standardize with Fashion-MNIST mean/std


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    raw = np.load("data/raw/fashion_mnist_raw.npz")
    x_train, y_train = normalize(raw["x_train"]), raw["y_train"]
    x_test, y_test = normalize(raw["x_test"]), raw["y_test"]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train,
    )

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/train.npz", x=x_tr, y=y_tr)
    np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
    np.savez_compressed("data/processed/test.npz", x=x_test, y=y_test)
    print(f"train={x_tr.shape}, val={x_val.shape}, test={x_test.shape}")


if __name__ == "__main__":
    main()