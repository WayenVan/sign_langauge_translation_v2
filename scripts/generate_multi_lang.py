from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from vllm.inputs.data import TextPrompt
from transformers import modeling_utils
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import polars

if (
    not hasattr(modeling_utils, "ALL_PARALLEL_STYLES")
    or modeling_utils.ALL_PARALLEL_STYLES is None
):
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

import datasets
from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["VLLM_USE_V1"] = "0"  # Use v1 API for vLLM
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

lang = "Chinese"
mname = "google/gemma-3-27b-it"
# mname = "sartifyllc/pawa-min-alpha"
working_dir = f"outputs/ph14t_{lang.lower()}"
output_dir = working_dir
lang = "zh"
data_root = "dataset/PHOENIX-2014-T-release-v3"

system = {
    "role": "system",
    "content": f"""You are a professional translator model that translates German to {lang}.
You should only show translation, do not genearte any function calls or tool calls. Do not add any additional prefixes or suffixes to the translation. The output should only inlucde {lang}. You should keep the details of the original query as much as possible, and do not change the meaning of the query.""",
}


os.makedirs(output_dir, exist_ok=True)

# ------------------- loading the model -------------------
model = LLM(
    model=mname,
    task="generate",
    model_impl="vllm",
    tensor_parallel_size=4,  # Adjust based on your GPU setup
    dtype="bfloat16",
    max_model_len=2048,
)

# ------------------- loading the tokenizer -------------------
tokenizer = AutoTokenizer.from_pretrained(mname)
print("----------------beginning of chat template----------------")
print(tokenizer.chat_template)
print("----------------end of chat template----------------")


# ------------------- preparing the processor -------------------
def do_translate(originals: list[str]):
    prompts = []
    for q in originals:
        query = [
            system,
            {
                "role": "user",
                "content": f"translate this to {lang} and only show the {lang} translation:  \n"
                + q
                + " /no_think \n",
            },
        ]
        prompts.append(
            dict(
                prompt=tokenizer.apply_chat_template(
                    query,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        )

    # beam_params = BeamSearchParams(
    #     beam_width=10,
    #     max_tokens=512,
    # )
    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=1024)
    outputs = model.generate(
        prompts,
        sampling_params=sampling_params,
    )
    return outputs


for subset in ["train", "dev", "test"]:
    # dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv
    anno_file = os.path.join(
        data_root,
        f"PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{subset}.corpus.csv",
    )
    output_file = os.path.join(output_dir, f"ph14t_{subset}_{lang}.csv")
    df = polars.read_csv(anno_file, separator="|", has_header=True)
    with open(output_file, "w") as f:
        f.write("name|translation\n")
        for row in df.iter_rows(named=True):
            name = row["name"]
            original = row["translation"]
            translated = do_translate([original])
            f.write(f"{name}|{translated[0].outputs[0].text}\n")
