import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union
import warnings

from transformers.cache_utils import Cache, DynamicCache
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
    is_flash_attn_2_available,
)

logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from .sustainablekv_utils import init_sustainablekv
from .cachekiller_utils import init_cachekiller

def mistral_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,  # [B, seq_len, 4096]
    attention_mask: Optional[torch.Tensor] = None,  # None
    position_ids: Optional[torch.LongTensor] = None,  # [B, seq_len]
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    # [SustainableKV] Initialize SustainableKV
    # init_sustainablekv(self)
    
    # [CacheKiller] Initialize CacheKiller
    init_cachekiller(self)

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, seq_len, 4096/H] = [1, 32, seq_len, 128]
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, kv_H, seq_len, 1024/kv_H] = [1, 8, seq_len, 128]
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, kv_H, seq_len, 1024/kv_H] = [1, 8, seq_len, 128]
    
    kv_seq_len = key_states.shape[-2]  # seq_len = length of prefix / 1 (after the first generation)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        # [SustainableKV] Add kv_seq_len in prepare_inputs_for_generation_mistral()
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            # vanilla implementation
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    # Because the input can be padded, the absolute sequence length depends on the max position id.
    rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, "sliding_window", None) is not None
        and kv_seq_len > self.config.sliding_window
    )
    
    if not _flash_supports_window_size:
        logger.warning_once(
            "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
            " make sure to upgrade flash-attn library."
        )
    
    # repeat k/v heads if n_kv_heads < n_heads
    # [SustainableKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)  # [B, H, seq_len, 1024/kv_H] = [1, 32, seq_len, 128]
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # [B, H, seq_len, 1024/kv_H] = [1, 32, seq_len, 128]

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        # second generation: past_key_value.key_cache[0].shape = [B, H, max_cache_size, head_dim]
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0

        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
        
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models

        # [SustainableKV] Add cache_killer
        if key_states.shape[-2] >= kv_seq_len:  # key_states.shape[-2] = q_len = sequence length / 1 (after the first generation)
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.cache_killer.update_kv(query_states, key_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)  # past_key_value.key_cache[0].shape = [B, H, max_cache_size, head_dim]
        else:
            # Vanilla implemetation
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        # if torch.is_autocast_enabled():
        #     target_dtype = torch.get_autocast_gpu_dtype()

        # Handle the case where the model is quantized
        if hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)
    
    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)  # second generation: [B, 1, H, head_dim]
    key_states = key_states.transpose(1, 2)  # second generation: [B, max_cache_size + 1, H, head_dim]
    value_states = value_states.transpose(1, 2)  # second generation: [B, max_cache_size + 1, H, head_dim]

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        use_sliding_windows=use_sliding_windows,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()  # second generation: [B, 1, 4096]
    attn_output = self.o_proj(attn_output)  # second generation: [B, 1, 4096]

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def prepare_inputs_for_generation_mistral(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    # [SustainableKV]
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            # usually will not get in
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()  # it is fixed to return None
        else:
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None
        
        # Keep only the unprocessed tokens:
        
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]
    
    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]
    
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    
    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )

    return model_inputs
