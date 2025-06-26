from torch import select_copy
from torch.utils.data import Dataset
import cv2
import numpy
import os
import polars as pl
from datasets import load_dataset
from functools import cached_property


class Ph14TGeneralDataset(Dataset):
    def __init__(self, data_root: str, mode: str = "train", pipline=None):
        self.data_root = data_root
        self.hg_dataset = load_dataset(
            os.path.join(os.path.dirname(__file__), "ph14t_builder.py"),
            data_dir=data_root,
            split=mode,
        )
        self.df = self.hg_dataset.to_polars()
        self.mode = mode

        self.pipline = pipline

    def __len__(self):
        return len(self.ids)

    @cached_property
    def ids(self):
        return self.hg_dataset.unique("name")

    def __getitem__(self, idx):
        id = self.ids[idx]
        data_info = self.get_data_info_by_id(id)

        video_frame_file_name = data_info["frame_files"]
        video_frame = []
        for frame_file in video_frame_file_name:
            image = cv2.imread(os.path.join(self.data_root, frame_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_frame.append(image)

        ret = dict(
            id=id,
            # NOTE: [time, height, width, channel], normalized to [0, 1]
            video=numpy.array(video_frame, dtype=numpy.float32) / 255.0,
            text=data_info["translation"],
        )

        if self.pipline:
            ret = self.pipline(ret)

        return ret

    def get_data_info_by_id(self, id: str):
        if id not in self.ids:
            raise ValueError(f"ID {id} not found in the dataset.")

        selected = self.df.filter(pl.col("name") == id)
        framefiles = selected.sort("frame_index")["frame_file"].to_list()

        if not framefiles:
            raise ValueError(f"No frame files found for ID {id}.")

        selected = selected.drop("frame_file", "frame_index").unique()

        assert len(selected) == 1, (
            f"Expected one entry for ID {id}, found {len(selected)}."
        )

        data_info = selected.to_dicts()[0]
        data_info["frame_files"] = [p["path"] for p in framefiles]

        return data_info


if __name__ == "__main__":
    data_root = "dataset/PHOENIX-2014-T-release-v3/"
    ph14t_dataset = Ph14TDataset(data_root, mode="train")
    print(f"Dataset size: {len(ph14t_dataset)}")
    for i in range(10):
        data_info = ph14t_dataset[i]
        print(
            f"ID: {data_info['id']}, Video shape: {data_info['video'].shape}, Text: {data_info['text']}"
        )
