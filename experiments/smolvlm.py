from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMModel
import logging
from transformers.models.smolvlm.processing_smolvlm import (
    SmolVLMProcessor,
    SmolVLMImagesKwargs,
)
import torch
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("outputs/smolvlm.log"))
logger.addHandler(logging.StreamHandler())

# from transformers.models.smolvlm.image_processing_smolvlm import (
#     SmolVLMImageProcessor,
#
# )
#
#

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")


messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "outputs/output.mp4"},
            {"type": "text", "text": "What is happening in this video?"},
        ],
    }
]


inputs = processor.apply_chat_template(
    [messages],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
)


# num = inputs.eq(49190).sum().item()  # Check if the input contains the expected token
# print(processor.batch_decode(inputs, skip_special_tokens=False))
# print(num)
# print(inputs)

model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

for name, param in model.named_parameters():
    if name.startswith("model.text_model"):
        info = f"{name}: {param.shape}"
        logger.info(info)


with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.model.vision_model(
            pixel_values=torch.randn(10, 3, 512, 512).to("cuda")
        )
        out = model.model.connector(out.last_hidden_state)
print(out.shape)
