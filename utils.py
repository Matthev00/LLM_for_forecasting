import torch
import numpy as np
import random
import argparse


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def parse_argument():
    parser = argparse.ArgumentParser(description="TimeLLM")
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
    parser.add_argument('--llm_layers', type=int, default=6)

    # optimization
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping patience"
    )
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--pct_start", type=float, default=0.2)
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
