from model.predictor import TimeLLM
from data.data_factory import data_provider
from engine import train
from utils import (
    set_seeds,
    parse_argument,
    load_content,
    save_model,
    create_writer,
    load_model,
)

import torch
from torch import nn

import os


def main():
    set_seeds()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_argument()
    args.content = load_content(args)
    args.num_workers = os.cpu_count()
    # model = TimeLLM(args).float()

    test_data, test_loader = data_provider(args, "test")
    X, y = next(iter(test_loader))
    print("X_shape:", X.shape)
    print("y_shape:", y.shape)

    # model = load_model(model, "time_llm_4_epochs.pth", device=device)
    # model.to(device)
    print(X)


if __name__ == "__main__":
    main()
