"""Stage 4 - evaluate the model on the test set, write metrics.json + confusion matrix."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    model = keras.models.load_model("models/model.h5")
    test = np.load("data/processed/test.npz")
    x, y = test["x"], test["y"]

    loss, acc = model.evaluate(x, y, verbose=0)
    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix(y, y_pred), display_labels=CLASSES).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig("plots/confusion_matrix.png", dpi=120)

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": round(float(loss), 4),
                   "test_accuracy": round(float(acc), 4)}, f, indent=2)
    print(f"test_loss={loss:.4f}  test_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()