from torch.utils.data import DataLoader
from typing import Tuple

from data_provider_pretrain.data_loader import Dataset_ETT_hour, Dataset_ETT_minute

data_dict = {"ETTh1": Dataset_ETT_hour}


def data_provider(args, flag: str = "train") -> Tuple[Dataset_ETT_minute, DataLoader]:
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    percent = args.percent

    if flag == "test":
        shuffle_flag = False
        drop_last = False
    else:
        shuffle_flag = True
        drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=(args.seq_len, args.label_len, args.pred_len),
        features=args.features,
        target=args.target,
        scale=args.scale,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
