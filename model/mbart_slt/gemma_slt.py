import logging


from einops import rearrange, einsum
from huggingface_hub import TextGenerationOutputFinishReason
import torch
from torch import nn, Tensor
from lightning import LightningModule
import os

from tensordict import TensorDict


from omegaconf import OmegaConf, DictConfig

from hydra.utils import instantiate

from typing import List
from transformers import get_scheduler
from torch import optim
from torch.optim import lr_scheduler as scheduler
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from torch.nn import functional as F
from misc.earth_mover_loss import masked_emd_batch
from misc.sign_cl import SignCL
from torch.nn.utils.rnn import pad_sequence

from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.models.gemma.tokenization_gemma_fast import Gemma3TokenizerFast
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config

from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers.modeling_outputs import BaseModelOutput
from torchmetrics import Accuracy
from torchmetrics.text import BLEUScore
from typing import Any
import copy
from misc.tuple_output import TupleOutput
from enum import Enum

from trl import AutoModelForSeq2SeqLMWithValueHead
# logger = logging.getLogger(__name__)


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class Gemma3SLT(LightningModule):
    MAX_TOKEN_LENGTH = 1024  # Maximum token length for MBart

    def __init__(self, cfg):
        super().__init__()

        self.cfg: DictConfig = cfg
        self.lang = "de_DE"  # Set source language to German
        self._init_mbart_model()

        self.visual_position_embedding = nn.Embedding(
            self.MAX_TOKEN_LENGTH, self.d_model
        )

        self.train_accu_visual = Accuracy(
            task="multiclass",
            num_classes=self.tokenizer.vocab_size,
            ignore_index=self.tokenizer.pad_token_id,
        )
        self.train_accu_text = Accuracy(
            task="multiclass",
            num_classes=self.tokenizer.vocab_size,
            ignore_index=self.tokenizer.pad_token_id,
        )
        self._init_visual_modules()

        self.bleu = BLEUScore(n_gram=1, smooth=True)
        self.blue4 = BLEUScore(n_gram=4, smooth=True)

        # self.connector = build_mlp(
        #     self.cfg.modules.connector_depth, self.d_model, self.d_model
        # )
        # self.text_connector = build_mlp(
        #     self.cfg.modules.connector_depth, self.d_model, self.d_model
        # )
        #
        # NOTE: some parameters which compatible with the original mbart model

    def _init_visual_modules(self):
        self.visual_backbone = instantiate(self.cfg.model.backbone)
        self.visual_adapter = instantiate(self.cfg.model.visual_adapter)

        self.visual_encoder_cfg = copy.deepcopy(self.mbart_config)
        self.visual_encoder_cfg.num_layers = (
            self.mbart_config.encoder_layers * self.cfg.model.visual_encoder_layer_scale
        )

        for param in self.visual_backbone.parameters():
            param.requires_grad = False
        self.visual_backbone.eval()

    def _init_gemma_model(self):
        mname = "google/gemma-3-4b-it"

        self.gemma = Gemma3ForCausalLM.from_pretrained(
            mname,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on TPUs
        )

        self.tokenizer = Gemma3TokenizerFast.from_pretrained(
            mname,
        )

        self.gemma_config = Gemma3Config.from_pretrained(mname)
        self.d_model = self.gemma_config.hidden_size

        self.start_video_id = self.tokenizer.convert_tokens_to_ids("<start_of_image>")

        self.start_video_embds = nn.Parameter(
            torch.zeros(1, self.d_model, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.end_video_embeds = nn.Parameter(
            torch.zeros(1, self.d_model, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )

        # freeze the mbart model except the decoder
        for param in self.mbart.parameters():
            param.requires_grad = False
        # for param in self.mbart.base_model.decoder.parameters():
        #     param.requires_grad = True
        for param in self.mbart.base_model.encoder.parameters():
            param.requires_grad = True

        # for name, param in self.mbart.base_model.decoder.named_parameters():
        #     if "self_attn" in name:
        #         param.requires_grad = False

        for param in self.mbart.base_model.shared.parameters():
            param.requires_grad = False
        for param in self.mbart.base_model.encoder.embed_positions.parameters():
            param.requires_grad = False
        for param in self.mbart.base_model.decoder.embed_positions.parameters():
            param.requires_grad = False

    def forward(
        self,
        video: Tensor,  # [B, T, C, H, W]
        video_length: Tensor,  # [B]
        text_input_ids: Tensor,  # [B, L]
        text_attention_mask: Tensor,  # [B, L]
        text_label_mask: Tensor | None = None,  # [B, L], optional label mask for text
    ):
        B = video_length.shape[0]
        input = self.prepare_for_casual_lm(
            text_input_ids,
            text_attention_mask,
            video,
            video_length,
            text_label_mask=text_label_mask,
        )
        labels = None
        if text_label_mask is not None:
            labels = torch.cat(
                [
                    input.input_ids[:, 1:],  # <bos>
                    torch.full(
                        (B, 1), self.tokenizer.eos_token_id, device=self.device
                    ),  # <eos>
                ]
            )
            labels = labels.masked_fill(
                input.label_mask == 0, -100
            )  # only preserve the text labels

        gemma_output = self.gemma.forward(
            attention_mask=input.attention_mask,  # [B, L]
            inputs_embeds=input.inputs_embeds,  # [B, L, D]
            labels=labels,  # [B, L]
        )
        return TupleOutput(
            gemma_output=gemma_output,  # Gemma3ForCausalLMOutput
            labels=labels,  # [B, L]
        )

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.
        """
        train_visual_acc = self.train_accu_visual.compute()
        train_text_acc = self.train_accu_text.compute()
        self.log(
            "train_generate_accu_visual",
            train_visual_acc,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_generate_accu_text", train_text_acc, prog_bar=True, sync_dist=True
        )

        self.train_accu_visual.reset()
        self.train_accu_text.reset()

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        """
        bleu = self.bleu.compute()
        self.log("val_generate_bleu", bleu, prog_bar=True, sync_dist=True)
        self.bleu.reset()

        bleu4 = self.blue4.compute()
        self.log("val_generate_bleu4", bleu4, prog_bar=True, sync_dist=True)
        self.blue4.reset()

    def prepare_for_casual_lm(
        self,
        text_input_ids: Tensor,  # [B, L] [<pad>, ..., <bos>, .... <start_of_image>, ...]
        text_attention_mask: Tensor,  # [B, L]
        video: Tensor,  # [BT, C, H, W]
        video_length: Tensor,  # [B], length of each video in the batch
        text_label_mask: Tensor | None = None,  # [B, L], optional label mask for text
    ):
        B, T, D = video_length.shape
        visual_output = self.get_visual_feats(video, video_length)

        input_ids = []
        attention_mask = []
        label_mask = []
        video_mask = []
        extened_visual_feats = []
        for b in range(B):
            video_token_pos = text_input_ids.ne(self.start_video_id).nonzero(
                as_tuple=True
            )[0]
            _input_ids = torch.cat(
                [
                    text_input_ids[b, :video_token_pos],  # before the first video token
                    torch.full(
                        T + 2, self.tokenizer.pad_token_id, device=input_ids.device
                    ),  # video tokens
                    text_input_ids[
                        b, video_token_pos + 1 :
                    ],  # after the first video token
                ]
            )
            _attention_mask = torch.cat(
                [
                    text_attention_mask[
                        b, :video_token_pos
                    ],  # before the first video token
                    torch.ones(1, device=input_ids.device),  # first video token
                    visual_output.attention_mask[b, :],  # video tokens
                    torch.ones(1, device=input_ids.device),
                    text_attention_mask[
                        b, video_token_pos + 1 :
                    ],  # after the first video token
                ]
            )
            if text_label_mask is not None:
                _label_mask = torch.cat(
                    [
                        text_label_mask[
                            b, :video_token_pos
                        ],  # before the first video token
                        torch.zeros(T + 2, device=input_ids.device),  # video tokens
                        text_label_mask[
                            b, video_token_pos + 1 :
                        ],  # after the first video token
                    ]
                )
                label_mask.append(_label_mask)
            _video_mask = torch.cat(
                [
                    torch.zeros(video_token_pos, device=input_ids.device),  # before
                    torch.ones(T + 2, device=input_ids.device),  # video tokens
                    torch.zeros(
                        text_input_ids.shape[1] - video_token_pos - 1,
                        device=input_ids.device,
                    ),  # after
                ]
            )
            _ex_visual_feat = torch.cat(
                [
                    torch.zeros(video_token_pos, D, device=input_ids.device),  # before
                    visual_output.visual_feats[b, :, :],  # video tokens
                    torch.zeros(
                        text_input_ids.shape[1] - video_token_pos - 1,
                        D,
                        device=input_ids.device,
                    ),  # after
                ]
            )
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            video_mask.append(video_mask)
            extened_visual_feats.append(_ex_visual_feat)

        text_embedding = self.mbart.get_input_embeddings()(
            torch.stack(input_ids, dim=0)
        )

        # replace the video token with the visual featurers
        inputs_embeds = text_embedding * (1 - video_mask.unsqueeze(-1)) + torch.stack(
            extened_visual_feats, dim=0
        )
        return TupleOutput(
            input_ids=torch.stack(input_ids, dim=0),  # [B, L]
            attention_mask=torch.stack(attention_mask, dim=0),  # [B, L]
            inputs_embeds=inputs_embeds,  # [B, L, D]
            label_mask=torch.stack(label_mask, dim=0)
            if text_label_mask is not None
            else None,  # [B, L]
            video_mask=torch.stack(video_mask, dim=0),  # [B, L]
        )

    def visual_position_embedding_forward(self, video_feats: Tensor):
        """
        Forward pass through the visual position embedding.
        args:
            video_feats: Tensor, shape [B, T, D], video features
        """
        B, T, D = video_feats.shape
        position_ids = (
            torch.arange(T, device=video_feats.device).unsqueeze(0).expand(B, -1)
        )
        position_embeddings = self.visual_position_embedding(position_ids)
        return video_feats + position_embeddings  # [B, T, D]

    def get_visual_feats(self, video: Tensor, video_length: Tensor):
        """
        Forward pass through the visual encoder.
        args:
            video: Tensor, shape [BT, C, H, W], concated video frames across batch
            video_length: Tensor, shape [B], length of each video in the batch
        """

        _, C, H, W = video.shape
        B = video_length.shape[0]

        video_feats = self.visual_backbone(video)  # [BT, CLS+HW+REGISTIRY, C]

        video_feats = torch.split(
            video_feats, video_length.tolist(), dim=0
        )  # list of [T, CLS+HW+REGISTIRY, C]
        video_feats = pad_sequence(
            list(video_feats), batch_first=True, padding_value=0.0
        ).contiguous()  # [B, T, CLS+HW+REGISTIRY, C]

        video_feats, video_length = self.visual_adapter(
            video_feats, video_length
        )  # [B, T, D]

        video_feats = self.visual_position_embedding_forward(video_feats)

        attention_mask = self.length_to_mask(video_length)

        return TupleOutput(
            visual_feats=video_feats,
            attention_mask=attention_mask,  # [B, 1+T]
        )

    @staticmethod
    def length_to_mask(lengths, max_length=None):
        """
        Convert lengths to a boolean mask.
        lengths: [B]
        max_length: int, optional
        """
        if max_length is None:
            max_length = lengths.max().item()
        B = lengths.size(0)
        mask = torch.arange(max_length, device=lengths.device).expand(
            B, max_length
        ) < lengths.unsqueeze(1)
        return mask.long()  # (B, max_length)

    def dispatch_batch(self, batch):
        video = batch["video"].to(self.device)  # [B, T, C, H, W]
        video_length = batch["video_length"].to(self.device)
        text_input_ids = batch["text_input_ids"].to(self.device)
        text_attention_mask = batch["text_attention_mask"].to(self.device)
        text_label_mask = (
            batch["text_label_mask"].to(self.device)
            if "text_label_mask" in batch
            else None
        )
        return TupleOutput(
            video=video,  # [B, T, C, H, W]
            video_length=video_length,  # [B]
            text_input_ids=text_input_ids,  # [B, L]
            text_attention_mask=text_attention_mask,  # [B, L]
            text_label_mask=text_label_mask,  # [B, L] or None
        )

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        batch = self.dispatch_batch(batch)

        assert batch.text_label_mask is not None, (
            "text_label_mask is required for training"
        )

        output = self.forward(
            video=batch.video,  # [B, T, C, H, W]
            video_length=batch.video_length,  # [B]
            text_input_ids=batch.text_input_ids,  # [B, L]
            text_attention_mask=batch.text_attention_mask,  # [B, L]
            text_label_mask=batch.text_label_mask,  # [B, L] or None
        )
        loss = output.gemma_output.loss  # Gemma3ForCausalLMOutput
        logits = output.gemma_output.logits  # [B, L, V]
        labels = output.labels  # [B, L]

        self.train_accu_visual.update(
            logits.view(-1, logits[:, : self.config.vocab_size].shape[-1]),  # [B*L, V]
            labels.view(-1),  # [B*L]
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        video_input, text_src_input, masked_text_src_input = self.dispatch_batch(batch)

        # text_encoder_out = self.text_encoder_forward(
        #     masked_text_src_input["input_ids"], masked_text_src_input["attention_mask"]
        # )

        visual_encoder_out = self.visual_encoder_forward(
            video_input["video"], video_input["video_length"]
        )

        tgt_text = self.tokenizer.batch_decode(
            text_src_input["input_ids"], skip_special_tokens=True
        )

        reference = [[t] for t in tgt_text]  # Reference for BLEU score

        output = self.mbart.generate(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=visual_encoder_out.visual_feats,
                # last_hidden_state=visual_global_feats.unsqueeze(1),  # [B, 1, D]
                hidden_states=None,
                attentions=None,
            ),
            attention_mask=visual_encoder_out.attention_mask,  # [B, T]
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang],
            num_beams=4,
            max_new_tokens=150,
        )

        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        self.bleu.update(decoded_output, reference)
        self.blue4.update(decoded_output, reference)

        return output

    def generate(
        self,
        video: Tensor,  # [B, T, C, H, W]
        video_length: Tensor,  # [B]
        lang=None,  # Language code for the target language
    ) -> List[str]:
        visual_encoder_out = self.visual_encoder_forward(video, video_length)

        output = self.mbart.generate(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=visual_encoder_out.visual_feats,
                # last_hidden_state=visual_global_feats.unsqueeze(1),  # [B, 1, D]
                hidden_states=None,
                attentions=None,
            ),
            attention_mask=visual_encoder_out.attention_mask,  # [B, T]
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang]
            if lang is None
            else self.tokenizer.lang_code_to_id[lang],
            num_beams=4,
            max_new_tokens=150,
        )

        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return decoded_output

    @staticmethod
    def contrastive_loss(
        visual_features: Tensor,  # [B, T, D]
        visual_mask: Tensor,  # [B, T]
        text_features: Tensor,  # [B, L, D]
        text_mask: Tensor,  # [B, L]
    ):
        """
        Compute contrastive loss between video and text embeddings.
        Args:
            visual_features: Video embeddings of shape (B, T, D)
            visual_mask: Attention mask for video of shape (B, T)
            text_features: Text embeddings of shape (B, L, D)
            text_mask: Attention mask for text of shape (B, L)
        """
        visual_feats = F.normalize(visual_features, dim=-1, p=2)
        text_feats = F.normalize(text_features, dim=-1, p=2).detach()

        similarity = einsum(
            visual_feats, text_feats, "b t d, b l d -> b t l"
        ).contiguous(())

        addictive_mask = text_mask.unsqueeze(1).float()
        addictive_mask = addictive_mask.masked_fill(addictive_mask == 0, float("-inf"))

        similarity = similarity + addictive_mask  # Apply padding mask

        values, index = similarity.max(dim=-1)  # Get max indices, (B, T)

        mean_values = values * visual_mask.float() / visual_mask.sum(-1, keepdim=True)

        return -mean_values.mean()  # Mean over batch

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.mbart.to(torch.bfloat16)  # Use bfloat16 for better performance on TPUs

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            filter(lambda p: p.requires_grad, self.parameters()),
            **self.cfg.engine.opt.kwargs,
        )
        # lr_scheduler, _ = create_scheduler_v2(
        #     optimizer,
        #     **self.cfg.engine.sched_encoder.kwargs,
        # )
        lr_scheduler = get_scheduler(
            self.cfg.engine.sched.name,
            optimizer=optimizer,
            **self.cfg.engine.sched.kwargs,
        )

        return [
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
        ]
