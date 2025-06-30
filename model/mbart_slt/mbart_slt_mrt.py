import logging


from einops import rearrange, einsum
from huggingface_hub import TextGenerationOutputFinishReason
import torch
from torch import nn, Tensor
from lightning import LightningModule
import os
from vector_quantize_pytorch import VectorQuantize

from tensordict import TensorDict


from omegaconf import OmegaConf, DictConfig

from sacrebleu import sentence_bleu
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

# from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack
# from transformers.models.t5.configuration_tize5 import T5Config
# from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.mbart50.tokenization_mbart50_fast import MBart50TokenizerFast
from transformers.models.mbart.modeling_mbart import (
    MBartEncoder,
    MBartForConditionalGeneration,
    shift_tokens_right,
)
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers.modeling_outputs import BaseModelOutput
from torchmetrics import Accuracy
from torchmetrics.text import BLEUScore
from typing import Any
import copy
from misc.tuple_output import TupleOutput

from .mbart_slt import MBartSLTModel

# logger = logging.getLogger(__name__)


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


MAX_TOKEN_LENGTH = 1024  # Maximum token length for MBart


class MBartSLTModelForMRT(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg: DictConfig = cfg

        self.base_model = MBartSLTModel(cfg)

        self.bleu = BLEUScore(n_gram=1, smooth=True)
        self.blue4 = BLEUScore(n_gram=4, smooth=True)

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
        video_input, text_src_input, masked_text_src_input = batch
        video_input = TensorDict(**video_input, device=self.device)
        text_src_input = TensorDict(**text_src_input, device=self.device)
        masked_text_src_input = TensorDict(**masked_text_src_input, device=self.device)
        return video_input, text_src_input, masked_text_src_input

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        video_input, text_src_input, masked_text_src_input = self.dispatch_batch(batch)

        B = video_input["video_length"].shape[0]

        visual_encoder_out = self.base_model.visual_encoder_forward(
            video_input["video"], video_input["video_length"]
        )

        all_canndidates = []
        for b in range(B):
            NUM_SAMPLES = 5
            cand = []
            for _ in range(NUM_SAMPLES):
                with torch.no_grad():
                    gennerated = self.base_model.mbart.generate(
                        encoder_outputs=BaseModelOutput(
                            last_hidden_state=visual_encoder_out.visual_feats[
                                b : b + 1
                            ],  # [1, T, D]
                            hidden_states=None,
                            attentions=None,
                        ),
                        attention_mask=visual_encoder_out.attention_mask[
                            b : b + 1
                        ],  # [1, T]
                        forced_bos_token_id=self.base_model.tokenizer.lang_code_to_id[
                            self.base_model.lang
                        ],
                        do_sample=True,
                        max_new_tokens=150,
                    )
                    cand.append(gennerated[0])
            all_canndidates.append(cand)

        rewards = torch.zeros(B, NUM_SAMPLES, device=self.device)
        for i in range(B):
            for j in range(NUM_SAMPLES):
                bleu = sentence_bleu(
                    self.base_model.tokenizer.decode(all_canndidates[i][j]),
                    [self.base_model.tokenizer.decode(text_src_input["input_ids"][i])],
                )
                rewards[i, j] = 1.0 - bleu.score / 100.0  # Convert BLEU to reward

        # NOTE: decoder forward for both visual and text
        # logits, _ = self.decoder_forward(visaul_connected_embeddings, decoder_input_ids)
        visual_logits, _ = self.decoder_forward(
            visual_encoder_out.visual_feats,  # [B, T, D]
            text_src_input["input_ids"],  # [B, L]
            visual_encoder_out.attention_mask,  # [B, T]
            text_src_input["attention_mask"],  # [B, L]
        )

        return total_loss

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
