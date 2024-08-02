import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import argparse
from pathlib import Path
import os
import datetime


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def parse_argument():
    parser = argparse.ArgumentParser(description="TimeLLM")
    # data
    parser.add_argument("--data", type=str, default="ETT", help="dataset type")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--freq", type=str, default="h", help="freq for time features encoding"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; "
        "M:multivariate predict multivariate, S: univariate predict univariate, "
        "MS:multivariate predict univariate",
    )
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    # model define
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--d_model", type=int, default=16, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--llm_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=32, help="dimension of fcn")
    parser.add_argument(
        "--llm_dim", type=int, default="768", help="LLM model dimension"
    )

    # optimization
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping patience"
    )
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--pct_start", type=float, default=0.2)
    parser.add_argument("--percent", type=int, default=100)

    args = parser.parse_args()
    return args


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epoch: int,
    args,
):
    new_lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def load_content(args):
    file = args.data
    with open("./data/prompt_bank/{0}.txt".format(file), "r") as f:
        content = f.read()
    return content


def save_model(model: torch.nn.Module, model_name:str):
    dir_path = Path("./model/pretrained")
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dir_path / model_name)


def test_data_loading(train_loader):
    print("Testing data loading...")
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}/{len(train_loader)} loaded.")
        if batch_idx >= 5:  # Testuj tylko kilka batchy
            break
    print("Data loading test completed.")


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> SummaryWriter:
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra) # noqa 5501
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    return SummaryWriter(log_dir=log_dir)

