import torch
from torch.utils.tensorboard import SummaryWriter  # noqa 5501

from utils import adjust_lr

from tqdm import tqdm
from typing import Tuple, Dict, List
from timeit import default_timer as timer


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
    writer: SummaryWriter = None,
) -> Dict[str, List[float]]:
    results = {
        "train_loss": [],
        "valid_loss": [],
        "mae_loss": [],
    }
    for epoch in range(epochs):
        start = timer()
        train_loss = train_step(
            args=args,
            model=model,
            dataloader=train_loader,
            optim=optim,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )
        adjust_lr(optim, scheduler, epoch, args)

        valid_loss, mae_loss = valid_step(
            args=args,
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            mae_metric=mae_metric,
            device=device,
        )

        results["train_loss"].append(train_loss)
        results["valid_loss"].append(valid_loss)
        results["mae_loss"].append(mae_loss)

        print(f"Epoch {epoch+1}/{epochs}\nTime: {timer()-start}")
        print(f"Train Loss: {train_loss}\nValid Loss: {valid_loss}")

        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": valid_loss},
                global_step=epoch,
            )

            writer.add_scalars(
                main_tag="MAE Loss",
                tag_scalar_dict={"mae_loss": mae_loss},
                global_step=epoch,
            )
    if writer:
        writer.close()
    return results


def train_step(
    args,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    criterion: torch.nn.Module,
    device: torch.device = "cuda:0",
) -> float:
    model.train()
    train_loss = 0.0
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
        enumerate(dataloader)
    ):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y[-1, :, :].float().to(device)

        outputs = model(batch_x)[0]

        f_dim = -1 if args.features == "MS" else 0
        outputs = outputs[-args.pred_len :, f_dim:]
        batch_y = batch_y[-args.pred_len :, f_dim:]

        loss = criterion(outputs, batch_y)
        train_loss += loss.item()

        optim.zero_grad()
        loss.backward
        optim.step()
    return train_loss / len(dataloader)


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

            outputs = model(batch_x)[0]

            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]
            batch_y = batch_y[:, -args.pred_len :, f_dim:]
            loss = criterion(outputs, batch_y)
            valid_loss += loss.item()
            mae_loss += mae_metric(outputs, batch_y).item()

    return valid_loss / len(valid_loader), mae_loss / len(valid_loader)
