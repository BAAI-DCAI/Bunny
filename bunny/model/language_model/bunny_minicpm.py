from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from bunny.model.language_model.minicpm.modeling_minicpm import MiniCPMModel, MiniCPMForCausalLM
from bunny.model.language_model.minicpm.configuration_minicpm import MiniCPMConfig

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM


class BunnyMiniCPMConfig(MiniCPMConfig):
    model_type = "bunny-minicpm"


class BunnyMiniCPMModel(BunnyMetaModel, MiniCPMModel):
    config_class = BunnyMiniCPMConfig

    def __init__(self, config: MiniCPMConfig):
        super(BunnyMiniCPMModel, self).__init__(config)


class BunnyMiniCPMForCausalLM(MiniCPMForCausalLM, BunnyMetaForCausalLM):
    config_class = BunnyMiniCPMConfig

    def __init__(self, config):
        super(MiniCPMForCausalLM, self).__init__(config)
        self.model = BunnyMiniCPMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
            if inputs_embeds is not None:
                inputs_embeds *= self.get_model().config.scale_emb

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("bunny-minicpm", BunnyMiniCPMConfig)
AutoModelForCausalLM.register(BunnyMiniCPMConfig, BunnyMiniCPMForCausalLM)
