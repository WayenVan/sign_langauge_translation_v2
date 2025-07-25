import numpy as np
import torch
from model.mbart_slt.constants import Language
import random


class Gemma3SLTMultilingCollator:
    """
    collator for my mbart model, which changes lots of things from the original gfslt collator

    """

    def __init__(
        self,
        tokenizer,
        mode="train",
    ):
        self.tokenizer = tokenizer
        self.mode = mode

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
        names, videos, texts, lang = (
            zbatch["id"],
            zbatch["augmented_video"],
            zbatch["text"],
            zbatch["lang"],
        )

        # Stack all videos into single tensor
        # video (T, C, H, W) ...
        #
        videos = [self.pad_dim_to_multiple_of_4(video, dim=0) for video in videos]
        video_lengths = [video.size(0) for video in videos]

        video_tensor = torch.cat(videos, dim=0).contiguous()
        video_lengths_tensor = torch.tensor(video_lengths)

        prompts = []
        n_labels = []
        text_input_ids = []
        text_input = []
        for i, t in enumerate(zbatch["text"]):
            if zbatch["lang"][i] == "en":
                message = [{"role": "user", "language": Language.EN.value}]
            elif zbatch["lang"][i] == "de":
                message = [{"role": "user", "language": Language.DE.value}]
            elif zbatch["lang"][i] == "zh":
                message = [{"role": "user", "language": Language.ZH.value}]
            else:
                raise ValueError(f"Unsupported language: {zbatch['language'][i]}")

            prompt_ids = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
            )
            prompts.append(self.tokenizer.decode(prompt_ids, skip_special_tokens=False))

            label_ids = self.tokenizer(t, add_special_tokens=False).input_ids

            if self.mode == "train":
                n_labels.append(len(label_ids))
                input_id = prompt_ids + label_ids
            else:
                # For inference, we only use the prompt
                n_labels.append(0)
                input_id = prompt_ids

            text_input_ids.append(input_id)
            text_input.append(
                self.tokenizer.decode(input_id, skip_special_tokens=False)
            )

        text_src_input = self.tokenizer.pad(
            {"input_ids": text_input_ids},
            padding=True,
            return_tensors="pt",  # or "tf" / "np"
        )

        # create the label mask
        if self.mode == "train":
            text_label_mask = torch.zeros_like(
                text_src_input.input_ids, dtype=torch.long
            )
            for i, n_label in enumerate(n_labels):
                # Set the mask to 0 for the label part
                text_label_mask[i, -n_label - 1 :] = 1
        else:
            text_label_mask = None

        # Prepare source input
        return {
            "video": video_tensor,
            "names": names,
            "video_length": video_lengths_tensor,
            "text_input": text_input,
            "text_input_ids": text_src_input.input_ids,
            "text_attention_mask": text_src_input.attention_mask,
            "text_label_mask": text_label_mask,
            "target_text": texts,
            "prompts": prompts,
            "lang": lang,
        }
