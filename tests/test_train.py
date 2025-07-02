import sys
import os
import logging

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # NOTE: this is the initial cwd when runing the sciprt

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lightning import Trainer
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import torch
from data.datamodule import DataModule

# from model.slt_vision_pretrain import SignBackboneForVPretraining
# from model.t5_text_pretrain import ModelForT5TextPretrain
# from model.mbart_slt import MBartSLTModel
import cv2

import datetime

from misc.git_utils import save_git_info
from typing import Any, Dict


def init_output_dir(file_name: str) -> str:
    """
    Initialize the output directory for the job.
    """
    now = datetime.datetime.now()
    subfolder = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", file_name, subfolder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def init_logger(local_rank, output_dir: str):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Stream handler for console output
            logging.FileHandler(
                os.path.join(output_dir, f"train_rank{local_rank}.log")
            ),  # File handler for logging to a file
        ],
    )
    return logging.getLogger(__name__)


# NOTE: the hydra appp only inisitalize once
@hydra.main(
    config_path="../configs",
    config_name="gfslt-vlp_pretrain_8a100",
    version_base="1.3.2",
)
def main(
    cfg: DictConfig,
) -> None:
    # NOTE: define callbacks for trainer
    cbs = [
        callbacks.RichProgressBar(),
        DebugCallback(),
    ]

    cfg.data.train.loader_kwargs.num_workers = 2

    # NOTE: start training
    t = Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        # devices=[2, 3, 4, 5, 6, 7],  # NOTE: specify the devices to use
        devices=[6, 7],
        callbacks=cbs,
        log_every_n_steps=cfg.log_interval,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=1.0,  # NOTE: gradient clipping will be normed
        # gradient_clip_algorithm="value",
        sync_batchnorm=True,
        precision=cfg.precision,
        logger=None,
        # WARN: will slow down the training process, just for debug now
        # detect_anomaly=True,
    )

    logger.info(f"Process in local rank {t.local_rank}, global rank {t.global_rank}")

    model = instantiate(cfg.model.type, cfg)
    datamodule = DataModule(cfg.data, model.tokenizer)
    t.fit(model, datamodule)


class DebugCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.current_train_batch = None

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.current_train_batch = batch
        self.logger = logging.getLogger("debug_callback")

    def on_before_backward(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        loss: torch.Tensor,
    ) -> None:
        # # NOTE: check the loss
        # if torch.isnan(loss).any():
        #     video = self.current_train_batch["video"]
        #     ids = self.current_train_batch["ids"]
        #
        #     self.logger.warning(f"Loss is NaN: {loss}")
        #     self.logger.warning(
        #         f"Video shape: {video.shape}, mean: {video.mean()}, std: {video.std()}"
        #     )
        #     self.logger.warning(f"input_ids: {ids}")
        # trainer.should_stop = True
        pass

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Any,
    ) -> None:
        nan_flag = False
        for name, param in pl_module.named_parameters():
            global_step = trainer.global_step
            if param.grad is None:
                logger.info(f"Parameter {name} has gradient None")
            else:
                logger.info(
                    f"Parameter {name} has grad mean: {param.grad.mean()}, std: {param.grad.std()}"
                )
        # if nan_flag and global_step >= 1000:
        #     logger.warning(
        #         "find nan and the global step is larger than 1000, stop the training"
        #     )
        #     trainer.should_stop = True
        return


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    filename = os.path.basename(__file__).split(".")[0]
    cv2.setNumThreads(
        0
    )  # NOTE: set the number of threads to 0 to avoid cv2 become over head

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))

    # NOTE: get or initialize the output directory
    output_dir = os.environ.get(
        filename.upper() + "_OUTPUT_DIR",
        None,
    )
    if output_dir is None:
        print(f"Output directory not found in environment variables, initializing...")
        output_dir = init_output_dir(filename)
        os.environ[filename.upper() + "_OUTPUT_DIR"] = output_dir

    logger = init_logger(local_rank, output_dir)
    main()
