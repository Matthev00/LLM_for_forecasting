from torch.utils.data import Dataset

from data.time_features import time_features
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class ETThour_Dataset(Dataset):
    def __init__(
        self,
        root_path: Path = Path("data/dataset/ETT-small"),
        flag: str = "train",
        size: Tuple[int, int, int] = None,
        features: str = "S",
        data_name: str = "ETTh1.csv",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        percent: int = 100,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.data_path = root_path / data_name
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        start_borders = [
            0,  # train_start
            12 * 30 * 24 - self.seq_len,  # val_start
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,  # test_start
        ]
        end_borders = [
            12 * 30 * 24,  # train_end
            12 * 30 * 24 + 4 * 30 * 24,  # val_end
            12 * 30 * 24 + 8 * 30 * 24,  # test_end
        ]

        start_border = start_borders[self.type]
        end_border = end_borders[self.type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[start_borders[0] : end_borders[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][start_border:end_border]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, axis=1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, axis=1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), axis=1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, axis=1)
            df_stamp = df_stamp.drop(["date"], axis=1)
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[start_border:end_border]
        self.data_y = data[start_border:end_border]
        self.data_stamp = data_stamp

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.seq_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scale:
            data = self.scaler.inverse_transform(data)
        return data
