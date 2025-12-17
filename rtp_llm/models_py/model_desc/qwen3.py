from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import FusedSiluActDenseMLP, RMSNorm
from rtp_llm.models_py.modules.attention import CausalAttention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


def print_pymodel_inputs(obj):
    print("=============print PyModelInputs")
    print(f"===={obj.input_ids.shape=}, {obj.input_ids.dtype=}")
    print(f"======{obj.attention_inputs.prefix_lengths.shape=},{obj.attention_inputs.prefix_lengths.dtype=}")
    print(f"======{obj.attention_inputs.sequence_lengths.shape=},{obj.attention_inputs.sequence_lengths.dtype=}")
    print(f"======{obj.attention_inputs.input_lengths.shape=},{obj.attention_inputs.input_lengths.dtype=}")
    if obj.attention_inputs.kv_cache_block_id_host is not None:
        print(f"======{obj.attention_inputs.kv_cache_block_id_host.shape=}, {obj.attention_inputs.kv_cache_block_id_host.dtype=}")
    else:
        print(f"======obj.attention_inputs.kv_cache_block_id_host is None")
    if obj.attention_inputs.kv_cache_block_id_device is not None:
        print(f"======{obj.attention_inputs.kv_cache_block_id_device.shape=}, {obj.attention_inputs.kv_cache_block_id_device.dtype=}")
    else:
        print(f"======obj.attention_inputs.kv_cache_block_id_device is None")
    print(f"======{obj.attention_inputs.is_prefill=}")
    if obj.attention_inputs.kv_block_offset is not None:
        print(f"======{obj.attention_inputs.kv_block_offset=}")
    else:
        print(f"======obj.attention_inputs.kv_block_offset is None")
    print(f"======{obj.attention_inputs.cu_seqlens.shape=}, {obj.attention_inputs.cu_seqlens.dtype=}")
    print(f"======{obj.attention_inputs.padding_offset.shape=}, {obj.attention_inputs.padding_offset.dtype=}")
    if obj.attention_inputs.cache_store_inputs is not None:
        print(f"======cache_store_inputs: {obj.attention_inputs.cache_store_inputs}")
    else:
        print(f"======cache_store_inputs: None")
    print(f"====bert_embedding_inputs:")
    print(f"========{obj.bert_embedding_inputs.combo_position_ids.shape=}, {obj.bert_embedding_inputs.combo_position_ids.dtype=}") if obj.bert_embedding_inputs.combo_position_ids is not None else print("========obj.bert_embedding_inputs.combo_position_ids is None")
    print(f"========{obj.bert_embedding_inputs.position_encoding.shape=}, {obj.bert_embedding_inputs.position_encoding.dtype=}") if obj.bert_embedding_inputs.position_encoding is not None else print("========obj.bert_embedding_inputs.position_encoding is None")
    print(f"========{obj.bert_embedding_inputs.combo_tokens_type_ids.shape=}, {obj.bert_embedding_inputs.combo_tokens_type_ids.dtype=}") if obj.bert_embedding_inputs.combo_tokens_type_ids is not None else print("========obj.bert_embedding_inputs.combo_tokens_type_ids is None")
    print(f"========{obj.bert_embedding_inputs.token_type_embedding.shape=}, {obj.bert_embedding_inputs.token_type_embedding.dtype=}") if obj.bert_embedding_inputs.token_type_embedding is not None else print("========obj.bert_embedding_inputs.token_type_embedding is None")
    print(f"========{obj.bert_embedding_inputs.input_embedding_scalar=}")


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.self_attn = CausalAttention(config, weights)
        self.mlp = FusedSiluActDenseMLP(config, weights)
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        print(f"=======Qwen3DecoderLayer forward: {residual.shape=}, {residual.dtype=}")
        hidden_states = self.input_layernorm(hidden_states)
        
        print(f"=======Qwen3DecoderLayer forward: {hidden_states.shape=}, {hidden_states.dtype=}")
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        print(f"=======Qwen3DecoderLayer forward after self attention: {hidden_states.shape=}, {hidden_states.dtype=}")
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        print(f"=======Qwen3DecoderLayer forward after ffn: {hidden_states.shape=}, {hidden_states.dtype=}")
        return hidden_states


class Qwen3Model(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)

        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, weights.weights[idx])
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        print_pymodel_inputs(inputs)
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        print(f"============={inputs_embeds.shape=}, {inputs_embeds.dtype=}")
        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        fmha_impl = self.get_fmha_impl(attention_inputs)
        print(f"=============Got fmha_impl: {fmha_impl.__class__.__name__}")
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            print(f"======Qwen3 forward decoder layer {i} done!!!======")
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Qwen3Model",
]
