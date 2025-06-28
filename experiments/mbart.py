from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration
from transformers.models.mbart50.tokenization_mbart50_fast import MBart50TokenizerFast

artical_de = "Der Generalsekretär der Vereinten Nationen sagt, dass es keine militärische Lösung in Syrien gibt."

model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt",
    device_map="auto",
)
tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)

# translate Arabic to English
tokenizer.src_lang = "de_DE"

# with tokenizer.as_target_tokenizer():
encoded_ar = tokenizer(artical_de, return_tensors="pt").to(model.device)


generated_tokens = model.generate(
    **encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"]
)
results = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "The Secretary-General of the United Nations says there is no military solution in Syria."


print(results)
