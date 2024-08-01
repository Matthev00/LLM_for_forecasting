from matplotlib import pyplot as plt
from typing import Dict, List


def plot_loss_curves(results: Dict[str, List[float]]) -> None:
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results["train_loss"], label="Training loss", color="blue")
    plt.plot(results["valid_loss"], label="Validation loss", color="red")
    plt.title("Loss curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(results["mae_loss"], label="MAE loss", color="green")
    plt.title("MAE loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main():
    results = {
        "train_loss": [0.1, 0.2, 0.3, 0.4, 0.5],
        "valid_loss": [0.2, 0.3, 0.4, 0.5, 0.6],
        "mae_loss": [0.3, 0.4, 0.5, 0.6, 0.7],
    }
    plot_loss_curves(results)


if __name__ == "__main__":
    main()