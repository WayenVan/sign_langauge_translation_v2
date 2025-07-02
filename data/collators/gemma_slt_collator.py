import numpy as np
import torch
from model.mbart_slt.constants import Language
import random


class Gemma3SLTCollator:
    """
    collator for my mbart model, which changes lots of things from the original gfslt collator

    """

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    @staticmethod
    def pad_dim_to_multiple_of_4(tensor, dim):
        current_size = tensor.size(dim)
        remainder = current_size % 4
        if remainder == 0:
            return tensor

        pad_size = 4 - remainder

        # 取这个维度的最后一个元素
        index = [slice(None)] * tensor.dim()
        index[dim] = -1
        last_element = tensor[tuple(index)].unsqueeze(dim)

        # 复制 pad_size 次
        padding = last_element.repeat_interleave(pad_size, dim=dim)
        return torch.cat([tensor, padding], dim=dim).contiguous()

    def __call__(self, batch):
        # Collate a batch of samples.
        zbatch = {key: tuple(dic[key] for dic in batch) for key in batch[0]}

        # Unpack batch data
        names, videos, texts = (
            zbatch["id"],
            zbatch["augmented_video"],
            zbatch["text"],
        )

        # Stack all videos into single tensor
        # video (T, C, H, W) ...
        #
        videos = [self.pad_dim_to_multiple_of_4(video, dim=0) for video in videos]
        video_lengths = [video.size(0) for video in videos]

        video_tensor = torch.cat(videos, dim=0).contiguous()
        video_lengths_tensor = torch.tensor(video_lengths)

        n_labels = []
        text_input_ids = []
        for t in zbatch["text"]:
            message = [{"role": "user", "language": Language.DE.value}]
            prompt_ids = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
            )
            label_ids = self.tokenizer(t, add_special_tokens=False).input_ids
            n_labels.append(len(label_ids))
            text_input_ids.append(prompt_ids + label_ids)

        text_src_input = self.tokenizer.pad(
            {"input_ids": text_input_ids},
            padding=True,
            return_tensors="pt",  # or "tf" / "np"
        )
        text_label_mask = torch.zeros_like(text_src_input.input_ids, dtype=torch.long)

        for i, n_label in enumerate(n_labels):
            # Set the mask to 0 for the label part
            text_label_mask[i, -n_label - 1 :] = 1

        # Prepare source input
        ret = {
            "video": video_tensor,
            "names": names,
            "video_length": video_lengths_tensor,
            "text_input_ids": text_src_input.input_ids,
            "text_attention_mask": text_src_input.attention_mask,
            "text_label_mask": text_label_mask,
        }
        return ret
