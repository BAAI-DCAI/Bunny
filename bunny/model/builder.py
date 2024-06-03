import os
import warnings
import torch

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, logging

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

from bunny.model import *


def load_pretrained_model(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", **kwargs):
    if model_type not in {'phi-1.5', 'phi-2', 'phi-3', 'stablelm-2', 'qwen1.5-1.8b', 'minicpm', 'llama3-8b'}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # Load Bunny model
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

        print('Loading Bunny from base model...')
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                         config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                            config=lora_cfg_pretrained,
                                                            **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Bunny weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')

            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        # this may be mm projector only
        print('Loading Bunny from base model...')

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                         config=cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                            **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)

        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    if getattr(model.config, "unfreeze_vision_tower", False):
        if 'lora' in model_name.lower():
            assert model_base is not None
            vision_non_lora_trainables = {k[19:]: v for k, v in non_lora_trainables.items() if
                                          k.startswith('model.vision_tower.')}
            vision_tower.load_state_dict(vision_non_lora_trainables, strict=False)
        else:
            assert model_base is None
            from safetensors.torch import load_file
            vision_weights = {}
            for file_name in os.listdir(model_path):
                if file_name.endswith('safetensors'):
                    vision_weights.update(
                        {k[19:]: v for k, v in load_file(os.path.join(model_path, file_name)).items() if
                         k.startswith('model.vision_tower.')})
            vision_tower.load_state_dict(vision_weights, strict=True)

    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model_type == 'llama3-8b':
        tokenizer.eos_token_id = 128001
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model, image_processor, context_len
