"""Stage 1 - download Fashion-MNIST and save raw arrays to data/raw/."""
import os

import numpy as np
from tensorflow import keras


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    os.makedirs("data/raw", exist_ok=True)
    np.savez_compressed(
        "data/raw/fashion_mnist_raw.npz",
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"Saved raw data: train={x_train.shape}, test={x_test.shape}")


if __name__ == "__main__":
    main()