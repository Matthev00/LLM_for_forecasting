import torch

from tqdm import tqdm


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
):
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, optim, scheduler, criterion, mae_metric)
        # Update scheduler
        valid_loss = valid_step(model, valid_loader, criterion, mae_metric)
        # print info


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
):
    pass
