import polars as pl
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from diskcache import Cache
import logging
from datasets import DatasetBuilder, DatasetInfo, SplitGenerator, Value
import datasets
from functools import cached_property

logger = logging.getLogger(__name__)


class Ph14TDatasetBuilder(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="PHOENIX-2014T dataset for huggingface datasets.",
            features=datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "translation": datasets.Value("string"),
                    "video": datasets.Value("string"),
                    "start": datasets.Value("int64"),
                    "end": datasets.Value("int64"),
                    "speaker": datasets.Value("string"),
                    "frame_file": datasets.Image(),
                    "frame_index": datasets.Value("int64"),
                }
            ),
        )

    @cached_property
    def df_train(self):
        return build_tables(self.config.data_dir, mode="train")

    @cached_property
    def df_validation(self):
        return build_tables(self.config.data_dir, mode="dev")

    @cached_property
    def df_test(self):
        return build_tables(self.config.data_dir, mode="test")

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"subset": self.df_train}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"subset": self.df_validation},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"subset": self.df_test}
            ),
        ]

    def _generate_examples(self, subset: pl.DataFrame):
        for idx, row in enumerate(subset.to_dicts()):
            yield idx, row


def build_tables(data_root: str, mode: str = "train"):
    raw_annotation = pl.read_csv(
        os.path.join(
            data_root,
            f"PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{mode}.corpus.csv",
        ),
        separator="|",
    )
    frame_file_table = generate_frame_file_table(
        data_root, get_feature_root(mode), raw_annotation["name"].to_list()
    )

    return raw_annotation.join(
        frame_file_table, left_on="name", right_on="id", how="left"
    )


def get_feature_root(mode: str = "train"):
    """
    Get the feature root directory for the specified mode.
    """
    return os.path.join("PHOENIX-2014-T/features/fullFrame-210x260px", mode)


def generate_frame_file_table(data_root, feature_root, ids):
    # Process files in parallel with dynamic chunking
    workers = min(32, (os.cpu_count() or 1) * 4)
    chunk_size = max(1, len(ids) // (workers * 4))  # Dynamic chunking

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Process batches of IDs in parallel
        futures = []
        for i in range(0, len(ids), chunk_size):
            batch = ids[i : i + chunk_size]
            futures.append(
                executor.submit(process_id_batch, data_root, feature_root, batch)
            )

        # Collect results with progress bar
        all_data = []
        for future in tqdm(futures, desc="Processing frames"):
            all_data.extend(future.result())

    # Single DataFrame creation
    return pl.DataFrame(
        all_data,
        schema={
            "id": pl.Utf8,
            "frame_file": pl.Utf8,
            "frame_index": pl.Int64,
        },
    )


def process_id_batch(data_root, feature_root, id_batch):
    batch_data = []
    for id in id_batch:
        for frame_file in get_video_frame_file_by_id(data_root, feature_root, id):
            batch_data.append(
                {
                    "id": id,
                    "frame_file": frame_file,
                    "frame_index": int(frame_file[-8:-4]),
                }
            )
    return batch_data


def get_video_frame_file_by_id(data_root: str, feature_root: str, id: str):
    """
    Get all frame files for a given video ID.
    """
    dir_path = os.path.join(feature_root, id)
    with os.scandir(os.path.join(data_root, dir_path)) as it:
        return [
            os.path.join(dir_path, entry.name)
            for entry in it
            if entry.name.endswith(".png") and entry.is_file()
        ]


if __name__ == "__main__":
    data_root = "dataset/PHOENIX-2014-T-release-v3/"
    d = datasets.load_dataset("data/ph14t/ph14t_builder.py", data_dir=data_root)
    print(d["train"].features)
    # table = build_tables(data_root, mode="train")
    # print(table)
