from model.predictor import TimeLLM
from data.data_factory import data_provider
from utils import set_seeds, parse_argument

import torch
from torch import nn


def main():
    set_seeds()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_argument()
    model = TimeLLM(args).float().to(device)

    train_data, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")
    train_steps = len(train_loader)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    optim = torch.optim.Adam(trained_parameters, lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate,
    )
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()


if __name__ == "__main__":
    main()
