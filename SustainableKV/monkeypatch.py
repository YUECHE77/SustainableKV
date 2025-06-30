from importlib.metadata import version
import warnings
import transformers

from .mistral_hijack import mistral_flash_attn2_forward as new_mistral_forward
from .mistral_hijack import prepare_inputs_for_generation_mistral as new_mistral_prepare_inputs

from .llama_hijack import llama_flash_attn2_forward as new_llama_forward
from .llama_hijack import prepare_inputs_for_generation_llama as new_llama_prepare_inputs

from .mixtral_hijack import mixtral_flash_attn2_forward as new_mixtral_forward
from .mixtral_hijack import prepare_inputs_for_generation_mixtral as new_mixtral_prepare_inputs

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def sustainablekv_replace_mistral():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")

    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = new_mistral_prepare_inputs
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = new_mistral_forward

def sustainablekv_replace_llama():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")

    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = new_llama_prepare_inputs
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = new_llama_forward

def sustainablekv_replace_mixtral():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    
    transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = new_mixtral_prepare_inputs
    transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = new_mixtral_forward
