import torch
from transformers import AutoTokenizer, AutoConfig, PreTrainedModel
from trl import (
    AutoModelForSeq2SeqLMWithValueHead,
    PPOTrainer,
    PPOConfig,
    create_reference_model,
)
import transformers


class VideoEncoderSeq2Seq(AutoModelForSeq2SeqLMWithValueHead):
    def forward(self, video_embeds, decoder_input_ids, attention_mask=None, **kwargs):
        # video_embeds: [batch, seq_v, dim_model]
        # 将 video_embeds 投影到 encoder 的 embedding 空间
        encoder_outputs = self.model.encoder(
            inputs_embeds=video_embeds, attention_mask=None
        )
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=None,
        )
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        value = self.value_head(decoder_outputs.last_hidden_state)  # scalar per token
        return transformers.modeling_outputs.Seq2SeqLMOutput(
            logits=logits,
            past_key_values=None,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_attentions=encoder_outputs.attentions,
            loss=None,
            value=value.squeeze(-1),
        )


# 加载 tokenizer 和 config
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
tokenizer.pad_token = tokenizer.eos_token

model = VideoEncoderSeq2Seq.from_pretrained("google/flan-t5-small")
ref_model = create_reference_model(model)  # frozen baseline

ppo_config = PPOConfig(
    batch_size=2,
    mini_batch_size=1,
    learning_rate=1e-5,
    max_prompt_length=0,
    max_completion_length=64,
    is_encoder_decoder=True,
)

ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

# 构造 dummy decoder inputs（如 BOS token）
batch_size = 2
decoder_input_ids = torch.full((batch_size, 1), tokenizer.bos_token_id)

# PPO 使用示例
video_embeds = torch.randn(batch_size, 16, model.config.d_model)  # example video dims
responses = ppo_trainer.generate(
    input_ids=None,
    video_embeds=video_embeds,
    decoder_input_ids=decoder_input_ids,
    max_new_tokens=32,
    do_sample=True,
    top_p=0.9,
)

# 计算 reward（自定义）
rewards = [torch.tensor(len(r)) for r in responses]

stats = ppo_trainer.step(
    queries=None,
    responses=responses,
    rewards=rewards,
    video_embeds=video_embeds,
    decoder_input_ids=decoder_input_ids,
)
print(stats)
