from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-27b-it",
)


# with open("jinjas/gemma_template_original.jinja", "w") as f:
#     f.write(tokenizer.chat_template)

with open("jinjas/gemma_slt.jinja", "r") as f:
    template = f.read()

tokenizer.chat_template = template


messages = [
    {
        "role": "user",
        "language": "English",
    },
]

gt = "Hello, how are you doing today?"

applied = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
)
# with tokenizer.as_target_tokenizer():
gt_ids = tokenizer(gt).input_ids

print("GT IDs:", gt_ids)


print(tokenizer.convert_ids_to_tokens(applied))
print(tokenizer.convert_ids_to_tokens(gt_ids))
