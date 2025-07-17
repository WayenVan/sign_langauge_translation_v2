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
from transformers import Gemma3ForConditionalGeneration, get_scheduler
from torch import optim
from torch.optim import lr_scheduler as scheduler
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from torch.nn import functional as F
from misc.earth_mover_loss import masked_emd_batch
from misc.sign_cl import SignCL
from torch.nn.utils.rnn import pad_sequence

from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from peft import get_peft_model, LoraConfig, TaskType

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
        self.random_video_mask = cfg.model.random_video_mask

        self.lora_config = cfg.model.lora_config

        self._init_gemma_model()

        self.visual_position_embedding = nn.Embedding(
            self.MAX_TOKEN_LENGTH, self.d_model
        )

        self.train_accu_visual = Accuracy(
            task="multiclass",
            num_classes=self.gemma_config.vocab_size,
            ignore_index=-100,
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
        self._post_init()

    @torch.no_grad()
    def _post_init(self):
        mean = self.gemma.get_input_embeddings().weight.data.mean(dim=0, keepdim=True)
        self.start_video_embds.copy_(mean)
        self.end_video_embeds.copy_(mean)

        # init the visual position embedding
        torch.nn.init.trunc_normal_(self.visual_position_embedding.weight, std=0.02)

    def _init_visual_modules(self):
        self.visual_backbone = instantiate(self.cfg.model.backbone)
        self.visual_adapter = instantiate(self.cfg.model.visual_adapter)

        if self.visual_backbone.is_lora:
            return  # lora model, no need to freeze the backbone
        else:
            for param in self.visual_backbone.parameters():
                param.requires_grad = False
            self.visual_backbone.eval()

    def _init_gemma_model(self):
        mname = "google/gemma-3-4b-it"

        gemma = Gemma3ForConditionalGeneration.from_pretrained(
            mname,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on TPUs
        )
        lora_config = LoraConfig(
            **self.lora_config,
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",  # 全注意力投影
                "gate_proj",
                "up_proj",
                "down_proj",  # GPT-style FFN 层
            ],
        )
        self.gemma = get_peft_model(gemma, lora_config)

        self.tokenizer = GemmaTokenizerFast.from_pretrained(
            mname,
        )
        with open(self.cfg.model.chat_template, "r") as f:
            self.tokenizer.chat_template = f.read()

        self.gemma_config = Gemma3Config.from_pretrained(mname).get_text_config()
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
        # for param in self.mbart.parameters():
        #     param.requires_grad = False
        # for param in self.mbart.base_model.decoder.parameters():
        #     param.requires_grad = True
        # for param in self.mbart.base_model.encoder.parameters():
        #     param.requires_grad = True

        # for name, param in self.mbart.base_model.decoder.named_parameters():
        #     if "self_attn" in name:
        #         param.requires_grad = False

        # for param in self.mbart.base_model.shared.parameters():
        #     param.requires_grad = False
        # for param in self.mbart.base_model.encoder.embed_positions.parameters():
        #     param.requires_grad = False
        # for param in self.mbart.base_model.decoder.embed_positions.parameters():
        #     param.requires_grad = Falsec
        self.gemma.disable_adapter()

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
                ],
                dim=1,
            )
            labels = labels.masked_fill(
                input.label_mask == 0, -100
            )  # only preserve the text labels

        gemma_output = self.gemma.forward(
            attention_mask=input.attention_mask,  # [B, L]
            inputs_embeds=input.inputs_embeds,  # [B, L, D]
        )
        return TupleOutput(
            gemma_output=gemma_output,  # Gemma3ForCausalLMOutput
            labels=labels,  # [B, L]
        )

    def generate(
        self,
        video: Tensor,  # [B, T, C, H, W]
        video_length: Tensor,  # [B]
        text_input_ids: Tensor,  # [B, L]
        text_attention_mask: Tensor,  # [B, L]
    ) -> List[str]:
        B = video_length.shape[0]

        input = self.prepare_for_casual_lm(
            text_input_ids,
            text_attention_mask,
            video,
            video_length,
            text_label_mask=None,
        )

        return self.gemma.generate(
            inputs_embeds=input.inputs_embeds,  # [B, L, D]
            attention_mask=input.attention_mask,  # [B, L]
            num_beams=4,
            max_new_tokens=150,
        )

    def prepare_for_casual_lm(
        self,
        text_input_ids: Tensor,  # [B, L] [<pad>, ..., <bos>, .... <start_of_image>, ...]
        text_attention_mask: Tensor,  # [B, L]
        video: Tensor,  # [BT, C, H, W]
        video_length: Tensor,  # [B], length of each video in the batch
        text_label_mask: Tensor | None = None,  # [B, L], optional label mask for text
    ):
        B = video_length.shape[0]
        visual_output = self.get_visual_feats(video, video_length)
        _, T, D = visual_output.visual_feats.shape

        input_ids = []
        attention_mask = []
        label_mask = []
        video_mask = []
        extened_visual_feats = []
        for b in range(B):
            video_token_pos = (
                text_input_ids[b].eq(self.start_video_id).nonzero(as_tuple=True)[0]
            )
            _input_ids = torch.cat(
                [
                    text_input_ids[b, :video_token_pos],  # before the first video token
                    torch.full(
                        (T + 2,), self.tokenizer.pad_token_id, device=self.device
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
                    torch.ones(1, device=self.device),  # first video token
                    visual_output.attention_mask[b, :],  # video tokens
                    torch.ones(1, device=self.device),
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
                        torch.zeros(T + 2, device=self.device),  # video tokens
                        text_label_mask[
                            b, video_token_pos + 1 :
                        ],  # after the first video token
                    ]
                )
                label_mask.append(_label_mask)

            _video_mask = torch.cat(
                [
                    torch.zeros(video_token_pos, device=self.device),  # before
                    torch.ones(T + 2, device=self.device),  # video tokens
                    torch.zeros(
                        text_input_ids.shape[1] - video_token_pos - 1,
                        device=self.device,
                    ),  # after
                ]
            )
            _ex_visual_feat = torch.cat(
                [
                    torch.zeros(video_token_pos, D, device=self.device),  # before
                    self.start_video_embds,
                    visual_output.visual_feats[b, :, :],  # video tokens
                    self.end_video_embeds,
                    torch.zeros(
                        text_input_ids.shape[1] - video_token_pos - 1,
                        D,
                        device=self.device,
                    ),  # after
                ]
            )
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            video_mask.append(_video_mask)
            extened_visual_feats.append(_ex_visual_feat)

        input_ids = torch.stack(input_ids, dim=0)  # [B, L]
        attention_mask = torch.stack(attention_mask, dim=0)  # [B, L]
        video_mask = torch.stack(video_mask, dim=0)  # [B, L]
        if text_label_mask is not None:
            label_mask = torch.stack(label_mask, dim=0)  # [B, L]
        extened_visual_feats = torch.stack(extened_visual_feats, dim=0)

        text_embedding = self.gemma.get_input_embeddings()(input_ids)

        # replace the video token with the visual featurers
        inputs_embeds = (
            text_embedding * (1 - video_mask.unsqueeze(-1)).float()
            + extened_visual_feats
        )

        if self.training and self.random_video_mask > 0.0:
            random_temporal_mask = torch.ones_like(
                attention_mask, device=self.device, dtype=torch.float32
            ) * (1.0 - self.random_video_mask)
            random_temporal_mask = random_temporal_mask.bernoulli()
            random_temporal_mask = random_temporal_mask.bool() | ~video_mask.bool()

            attention_mask = attention_mask * random_temporal_mask.long()

        return TupleOutput(
            input_ids=input_ids,  # [B, L]
            attention_mask=attention_mask,  # [B, L]
            inputs_embeds=inputs_embeds,  # [B, L, D]
            label_mask=label_mask if text_label_mask is not None else None,  # [B, L]
            video_mask=video_mask,  # [B, L]
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
        return TensorDict(batch).to(self.device)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        batch = self.dispatch_batch(batch)

        assert batch["text_label_mask"] is not None, (
            "text_label_mask is required for training"
        )

        output = self.forward(
            video=batch["video"],  # [B, T, C, H, W]
            video_length=batch["video_length"],  # [B]
            text_input_ids=batch["text_input_ids"],  # [B, L]
            text_attention_mask=batch["text_attention_mask"],  # [B, L]
            text_label_mask=batch["text_label_mask"],  # [B, L] or None
        )

        logits = output.gemma_output.logits  # [B, L, V]
        labels = output.labels  # [B, L]

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),  # [B*L, V]
            labels.view(-1),  # [B*L]
            ignore_index=-100,  # ignore the padding tokens
        )

        self.train_accu_visual.update(
            logits.view(-1, logits.shape[-1]),  # [B*L, V]
            labels.view(-1),  # [B*L]
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.
        """
        train_visual_acc = self.train_accu_visual.compute()
        self.log(
            "train_generate_accu_visual",
            train_visual_acc,
            prog_bar=True,
            sync_dist=True,
        )

        self.train_accu_visual.reset()

    def validation_step(self, batch, batch_idx):
        batch = self.dispatch_batch(batch)

        output = self.generate(
            video=batch["video"],  # [B, T, C, H, W]
            video_length=batch["video_length"],  # [B]
            text_input_ids=batch["text_input_ids"],  # [B, L]
            text_attention_mask=batch["text_attention_mask"],  # [B, L]
        )

        reference = [[t] for t in batch["target_text"]]  # [B, 1] list of lists

        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        self.bleu.update(decoded_output, reference)
        self.blue4.update(decoded_output, reference)

        return output

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

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.gemma.base_model.to(
            torch.bfloat16
        )  # Use bfloat16 for better performance on TPUs

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
