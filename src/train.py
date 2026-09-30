"""Stage 3 - build and train the ANN, save models/model.h5 and models/history.csv."""
import os

import numpy as np
import pandas as pd
import yaml
from tensorflow import keras


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    keras.utils.set_random_seed(p["seed"])

    train = np.load("data/processed/train.npz")
    val = np.load("data/processed/val.npz")

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train["x"], train["y"],
        validation_data=(val["x"], val["y"]),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        verbose=2,
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
    print("Saved models/model.h5 and models/history.csv")


if __name__ == "__main__":
    main()