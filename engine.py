import torch
from torch.utils.tensorboard import SummaryWriter  # noqa 5501

from tqdm import tqdm
from typing import Tuple, Dict, List


def train(
    args,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    criterion: torch.nn.Module,
    mae_metric: torch.nn.Module,
    epochs: int = 10,
    device: torch.device = "cuda:0",
) -> Dict[str, List[float]]:
    results = {
        "train_loss": [],
        "valid_loss": [],
    }
    for epoch in range(epochs):
        train_loss = train_step(
            model, train_loader, optim, scheduler, criterion, mae_metric
        )
        # Update scheduler
        valid_loss = valid_step(model, valid_loader, criterion, mae_metric)

        results["train_loss"].append(train_loss)
        results["valid_loss"].append(valid_loss)
        # print info

        # add tensorboard writer
    return results


def train_step(
    args,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    criterion: torch.nn.Module,
    device: torch.device = "cuda:0",
) -> float:
    model.train()
    train_loss = 0.0
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
        enumerate(train_loader)
    ):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(device)
        dec_inp = (
            torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
            .float()
            .to(device)
        )

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

        f_dim = -1 if args.features == "MS" else 0
        outputs = outputs[:, -args.pred_len :, f_dim:]
        batch_y = batch_y[:, -args.pred_len :, f_dim:]
        loss = criterion(outputs, batch_y)
        train_loss += loss.item()

        optim.zero_grad()
        loss.backward
        optim.step()
    return train_loss / len(train_loader)


def valid_step(
    args,
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    mae_metric: torch.nn.Module,
    device: torch.device = "cuda:0",
) -> Tuple[float, float]:
    model.eval()
    valid_loss, mae_loss = 0.0, 0.0

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(valid_loader)
        ):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = (
                torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(device)
            )
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(device)
            )

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]
            batch_y = batch_y[:, -args.pred_len :, f_dim:]
            loss = criterion(outputs, batch_y)
            valid_loss += loss.item()
            mae_loss += mae_metric(outputs, batch_y).item()

    return valid_loss / len(valid_loader), mae_loss / len(valid_loader)
