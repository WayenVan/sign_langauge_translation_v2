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

from transformers.models.mbart50.tokenization_mbart50_fast import MBart50TokenizerFast
from transformers.models.mbart.modeling_mbart import (
    MBartEncoder,
    MBartForConditionalGeneration,
    shift_tokens_right,
)
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel


from transformers.generation.utils import GenerationMixin
from torchmetrics import Accuracy
from torchmetrics.text import BLEUScore
from typing import Any
import copy
from misc.tuple_output import TupleOutput

from trl.models.modeling_value_head import ValueHead
# logger = logging.getLogger(__name__)


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


MAX_TOKEN_LENGTH = 1024  # Maximum token length for MBart


class MBartSLTModel(PreTrainedModel, GenerationMixin, LightningModule):
    def __init__(self, cfg):
        self.mname = "facebook/mbart-large-50-many-to-many-mmt"
        self.mbart_config = MBartConfig.from_pretrained(self.mname)
        super().__init__(self.mbart_config)

        self.cfg: DictConfig = cfg

        self.lang = "de_DE"  # Set source language to German
        self._init_mbart_model()

        self.visual_position_embedding = nn.Embedding(MAX_TOKEN_LENGTH, self.d_model)

        self.visual_src_lang_token = nn.Parameter(
            torch.randn(1, 1, self.d_model),
            requires_grad=True,
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
        # set up the pretrained model parameters
        self.is_peft_model = False

    def _init_visual_modules(self):
        self.visual_backbone = instantiate(self.cfg.model.backbone)
        self.visual_adapter = instantiate(self.cfg.model.visual_adapter)

        self.visual_encoder_cfg = copy.deepcopy(self.mbart_config)
        self.visual_encoder_cfg.num_layers = (
            self.mbart_config.encoder_layers * self.cfg.model.visual_encoder_layer_scale
        )
        self.visual_encoder = MBartEncoder(self.visual_encoder_cfg)

        for param in self.visual_backbone.parameters():
            param.requires_grad = False
        self.visual_backbone.eval()

    def _init_mbart_model(self):
        self.mbart = MBartForConditionalGeneration.from_pretrained(
            self.mname,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on TPUs
        )

        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            self.mname,
        )
        self.tokenizer.src_lang = self.lang

        self.d_model = self.mbart_config.d_model

        self.eos_token = "</s>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)

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

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Any,
    ):
        """
        a dummy generate function to make the model compatible with the trl library.
        input_ids is the concated video frames across batch,
        attention_mask is the length of each video in the batch.
        """
        video = input_ids  # Assuming input_ids is the video tensor
        video_length = attention_mask  # Assuming attention_mask is the video length

        visual_encoder_out = self.visual_encoder_forward(video, video_length)

        return self.mbart.generate(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=visual_encoder_out.visual_feats,
                # last_hidden_state=visual_global_feats.unsqueeze(1),  # [B, 1, D]
                hidden_states=None,
                attentions=None,
            ),
            attention_mask=visual_encoder_out.attention_mask,  # [B, T]
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang],
            # num_beams=4,
            # max_new_tokens=150,
            **kwargs,  # Pass any additional arguments to the generate method
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        video = input_ids  # Assuming input_ids is the video tensor
        video_length = attention_mask  # Assuming attention_mask is the video length

        visual_encoder_out = self.visual_encoder_forward(video, video_length)

        return self.mbart.forward(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=visual_encoder_out.visual_feats,
                # last_hidden_state=visual_global_feats.unsqueeze(1),  # [B, 1, D]
                hidden_states=None,
                attentions=None,
            ),
            attention_mask=visual_encoder_out.attention_mask,  # [B, T]
            **kwargs,  # Pass any additional arguments to the decoder
        )

    def get_eos_embedding(self):
        """
        Get the embedding for the end-of-sequence token.
        """
        eos_token_id = self.eos_token_id
        eos_embedding = self.mbart.get_input_embeddings()(
            torch.tensor([eos_token_id])
        ).unsqueeze(0)  # [1, 1, D]
        return eos_embedding

    def get_bos_embedding(self):
        """
        Get the embedding for the beginning-of-sequence token.
        """
        bos_token_id = self.bos_token_id
        bos_embedding = self.mbart.get_input_embeddings()(
            torch.tensor([bos_token_id])
        ).unsqueeze(0)  # [1, 1, D]
        return bos_embedding

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

    def visual_encoder_forward(self, video: Tensor, video_length: Tensor):
        """
        Forward pass through the visual encoder.
        args:
            video: Tensor, shape [BT, C, H, W], concated video frames across batch
            video_length: Tensor, shape [B], length of each video in the batch
        """

        _, C, H, W = video.shape
        B = video_length.shape[0]

        video_feats = self.visual_backbone(video)  # [BT, H, W, C]

        video_feats = torch.split(
            video_feats, video_length.tolist(), dim=0
        )  # list of [T, C, H, W]
        video_feats = pad_sequence(
            list(video_feats), batch_first=True, padding_value=0.0
        ).contiguous()  # [B, T, C, H, W]

        video_feats, video_length = self.visual_adapter(
            video_feats, video_length
        )  # [B, T, D]

        video_feats = self.visual_position_embedding_forward(video_feats)

        visual_src_lang_token = self.visual_src_lang_token.expand(B, 1, -1)

        inputs_embeds = torch.cat(
            [
                visual_src_lang_token,
                video_feats,  # [B, T, D]
            ],
            dim=1,
        ).contiguous()  # [B, 1+T, D]

        video_length = video_length + 1
        attention_mask = self.length_to_mask(video_length)

        encoder_outputs = self.visual_encoder(
            inputs_embeds=inputs_embeds,  # [B, T, D]
            attention_mask=attention_mask,  # [B, T]
            output_hidden_states=True,  # Enable hidden states output
        )

        visual_last_hidden_states = encoder_outputs.last_hidden_state  # [B, T, D]

        visual_feats = visual_last_hidden_states  # [B, T, D]
        visual_lang_feats = visual_feats[:, 0, :]  # [B, D] (global token)

        return TupleOutput(
            visual_feats=visual_feats,  # [B, 1+T, D]
            attention_mask=attention_mask,  # [B, 1+T]
            video_feats=video_feats,  # [B, T, D]
            video_length=video_length,  # [B]
            visual_lang_feats=visual_lang_feats,  # [B, D]
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

    def text_encoder_forward(self, input_ids: Tensor, attention_mask: Tensor):
        B, _ = input_ids.shape

        text_embeddings = self.mbart.get_input_embeddings()(input_ids)
        inputs_embeds = text_embeddings

        encoder_outputs = self.mbart.base_model.encoder(
            # input_ids=input_ids,  # [B, L, D]
            inputs_embeds=inputs_embeds,  # [B, L, D]
            attention_mask=attention_mask,  # [B, L]
        )

        text_last_hidden_states = encoder_outputs.last_hidden_state  # [B, L, D]
        # text_feats = self.text_connector(text_last_hidden_states)
        #
        text_feats = text_last_hidden_states  # [B, L, D]

        return TupleOutput(
            text_feats=text_feats,  # [B, L, D]
            attention_mask=attention_mask,  # [B, L]
            text_lang_feats=text_feats[:, 0, :],  # [B, D] (global token)
        )

    def decoder_forward(
        self,
        encoder_hidden_states,  # [B, G, D]
        decoder_input_ids: Tensor,  # [B, L]
        encoder_attn_mask: Tensor
        | None = None,  # [B, L] (optional, not used in this case)
        decoder_attn_mask: Tensor
        | None = None,  # [B, L] (optional, not used in this case)
    ):
        """
        Forward pass through the T5 decoder.
        """
        decoder_input_ids = shift_tokens_right(
            decoder_input_ids,  # [B, L]
            self.tokenizer.pad_token_id,  # Padding token ID
        )
        decoder_input_embeddings = self.mbart.get_input_embeddings()(
            decoder_input_ids
        )  # [B, L, D]
        decoder_outputs = self.mbart.base_model.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attn_mask,  # [B, T] (optional, not used in this case)
            use_cache=False,  # Disable cache for training
        )
        logits = self.mbart.lm_head(decoder_outputs.last_hidden_state)  # [B, L, C]

        return TupleOutput(
            logits=logits,  # [B, L, C]
            last_hidden_state=decoder_outputs.last_hidden_state,  # [B, L, D]
        )

    def dispatch_batch(self, batch):
        video_input, text_src_input, masked_text_src_input = batch
        video_input = TensorDict(**video_input, device=self.device)
        text_src_input = TensorDict(**text_src_input, device=self.device)
        masked_text_src_input = TensorDict(**masked_text_src_input, device=self.device)
        return video_input, text_src_input, masked_text_src_input

    def visual_text_loss(
        self,
        visual_embedding: Tensor,  # [B,  D]
        text_embedding: Tensor,  # [B, D]
    ):
        """
        Calculate the loss for the visual-text alignment.
        This is a placeholder method and should be implemented based on the specific task.
        """
        # first try simple mse
        text_embedding = text_embedding

        text_embedding = F.normalize(text_embedding, dim=-1)
        visual_embedding = F.normalize(visual_embedding, dim=-1)

        mse = (
            F.mse_loss(visual_embedding, text_embedding, reduction="none")
            .sum(dim=-1)
            .mean()
        )
        return mse

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        video_input, text_src_input, masked_text_src_input = self.dispatch_batch(batch)

        B = video_input["video_length"].shape[0]

        text_encoder_out = self.text_encoder_forward(
            masked_text_src_input["input_ids"], masked_text_src_input["attention_mask"]
        )

        visual_encoder_out = self.visual_encoder_forward(
            video_input["video"], video_input["video_length"]
        )

        # NOTE: decoder forward for both visual and text
        # logits, _ = self.decoder_forward(visaul_connected_embeddings, decoder_input_ids)
        visual_logits, _ = self.decoder_forward(
            visual_encoder_out.visual_feats,  # [B, T, D]
            text_src_input["input_ids"],  # [B, L]
            visual_encoder_out.attention_mask,  # [B, T]
            text_src_input["attention_mask"],  # [B, L]
        )
        text_logits, _ = self.decoder_forward(
            text_encoder_out.text_feats,  # [B, L, D]
            text_src_input["input_ids"],  # [B, L]
            text_encoder_out.attention_mask,  # [B, L]
            text_src_input["attention_mask"],  # [B, L]
        )

        # Calculate visual-text loss
        visual_text_loss = (
            self.visual_text_loss(
                visual_encoder_out.visual_lang_feats,  # [B, D]
                text_encoder_out.text_lang_feats,  # [B, D]
            )  # [B, D])
            * self.cfg.visual_text_alignment_weight
        )

        self.log(
            "train_visual_text_loss",
            visual_text_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )

        # Calculate visual generate loss
        # LABEL_LENGTH = labels.shape[1]
        generate_loss_visual = F.cross_entropy(
            rearrange(visual_logits, "b l c -> (b l) c").contiguous(),
            rearrange(text_src_input["input_ids"], "b l -> (b l)"),
            ignore_index=self.tokenizer.pad_token_id,
        )
        self.log(
            "train_generate_loss_visual",
            generate_loss_visual,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )

        # calculate text generate loss
        generate_loss_text = F.cross_entropy(
            rearrange(text_logits, "b l c -> (b l) c").contiguous(),
            rearrange(text_src_input["input_ids"], "b l -> (b l)"),
            ignore_index=-100,
        )
        self.log(
            "train_generate_loss_text",
            generate_loss_text,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )

        # Update accuracy
        self.train_accu_visual.update(
            rearrange(visual_logits, "b l c -> (b l) c")[
                :, : self.tokenizer.vocab_size
            ],
            rearrange(text_src_input["input_ids"], "b l -> (b l)"),
        )
        self.train_accu_text.update(
            rearrange(text_logits, "b l c -> (b l) c")[:, : self.tokenizer.vocab_size],
            rearrange(text_src_input["input_ids"], "b l -> (b l)"),
        )

        total_loss = (
            visual_text_loss + generate_loss_visual + generate_loss_text
            # + signcl_loss
        )
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
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
