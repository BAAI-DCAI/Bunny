# Copyright 2023 Stability and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" StableLM Epoch model configuration"""
from transformers import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class StableLMEpochConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50_304):
            Vocabulary size of the StableLM model. Defines the number of different tokens that
            can be represented by the `inputs_ids` passed when calling [`StableLMEpochModel`].
        intermediate_size (`int`, *optional*, defaults to 6912):
            Dimension of the MLP representations.
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the decoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string).
        rope_pct (`float`, *optional*, defaults to 1.0):
            Percentage of hidden dimensions to allocate to rotary embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 1e-5):
            The standard deviation of the truncated_normal_initializer for initializing
             all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-8):
            The epsilon used by the normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions
            (not used by all models). Only relevant if `config.is_decoder=True`.
        use_qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use bias for qkv layers.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
    """
    model_type = "stablelm_epoch"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50_304,
        intermediate_size=6912,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        rope_pct=0.25,
        rope_theta=10_000,
        max_position_embeddings=4096,
        initializer_range=0.02,
        norm_eps=1.0e-5,
        use_cache=True,
        use_qkv_bias=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rope_pct = rope_pct
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.use_qkv_bias = use_qkv_bias
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
