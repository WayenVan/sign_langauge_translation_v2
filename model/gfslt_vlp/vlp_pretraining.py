from lightning import LightningModule
from transformers.models.mbart.tokenization_mbart_fast import MBartTokenizerFast
from torch import nn
from .modeling_vlp import SLRCLIP, Text_Decoder
from .utils import KLLoss
from .optim import create_optimizer, create_scheduler
from torch.optim import AdamW
from torch.optim import lr_scheduler as scheduler

import torch


class VLPPretraining(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = MBartTokenizerFast.from_pretrained(cfg.model.tokenizer)
        self.pad_idx = self.tokenizer.pad_token_id

        self.model = SLRCLIP(cfg.model)
        self.text_decoder = Text_Decoder(cfg.model)

        # Loss functions
        self.loss_img = KLLoss()
        self.loss_txt = KLLoss()
        self.loss_fct = nn.CrossEntropyLoss(
            ignore_index=self.pad_idx, label_smoothing=0.2
        )
        self.n_text_decoder_forward_pass = self.cfg.model.n_text_decoder_forward_pass

    def training_step(self, batch, batch_idx):
        src_input, tgt_input, masked_tgt_input = batch

        # Main model forward pass
        logits_per_image, logits_per_text, ground_truth = self.model(
            src_input, tgt_input
        )
        loss_imgs = self.loss_img(logits_per_image.log_softmax(dim=-1), ground_truth)
        loss_texts = self.loss_txt(logits_per_text.log_softmax(dim=-1), ground_truth)
        total_loss = (loss_imgs + loss_texts) / 2.0

        # Text decoder forward pass (every n steps)
        if batch_idx % self.n_text_decoder_forward_pass == 0:
            lm_logits = self.text_decoder(
                tgt_input, masked_tgt_input, self.model.model_txt
            )
            masked_lm_loss = (
                self.loss_fct(
                    lm_logits.view(-1, lm_logits.shape[-1]),
                    tgt_input["input_ids"].view(-1),
                )
                * self.hparams.args.loss_lambda
            )
            total_loss += masked_lm_loss

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_img = self.loss_img
        loss_txt = self.loss_txt
        loss_fct = self.loss_fct

        src_input, tgt_input, masked_tgt_input = batch

        with torch.no_grad():
            logits_per_image, logits_per_text, ground_truth = self.model(
                src_input, tgt_input
            )
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)

            lm_logits = self.text_decoder(
                tgt_input, masked_tgt_input, self.model.model_txt
            )
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, lm_logits.shape[-1]),
                tgt_input["input_ids"].cuda().view(-1),
            )
            total_loss = (loss_imgs + loss_texts) / 2.0

        self.log(
            "total_val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        # Main model optimizer
        optimizer = create_optimizer(self.cfg.engine.opt_args, self.model)
        lr_scheduler, _ = create_scheduler(self.cfg.engine.sched_args, optimizer)

        # Text decoder optimizer
        optimizer_td = AdamW(
            self.text_decoder.parameters(),
            lr=self.cfg.engine.lr_opt_td,
            weight_decay=0,
            betas=(0.9, 0.98),
        )
        lr_scheduler_td = scheduler.CosineAnnealingLR(
            optimizer=optimizer_td,
            eta_min=1e-8,
            T_max=self.cfg.max_epochs,
        )

        return [
            {"optimizer": optimizer, "lr_scheduler": lr_scheduler},
            {"optimizer": optimizer_td, "lr_scheduler": lr_scheduler_td},
        ]
