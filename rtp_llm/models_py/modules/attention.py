from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_reduce
from rtp_llm.models_py.modules import FusedQKRMSNorm
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.ops import KVCache
from rtp_llm.utils.model_weight import W


class CausalAttention(nn.Module):

    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.head_num
        self.head_num = config.head_num
        self.num_key_value_groups = config.head_num // config.head_num_kv
        self.q_size = config.head_num * self.head_dim
        # Create linear layers using LinearFactory
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_qkv_w, W.attn_qkv_s, W.attn_qkv_b, config
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_o_w, W.attn_o_s, W.attn_o_b, config
        )
        self.qk_fuse_norm = None
        if W.q_ln_gamma in weights and W.k_ln_gamma in weights:
            self.qk_fuse_norm = FusedQKRMSNorm(
                weights[W.q_ln_gamma],
                weights[W.k_ln_gamma],
                config.head_num // config.tp_size,
                config.head_num_kv // config.tp_size,
                config.size_per_head,
                config.layernorm_eps,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        print(f"=======CasualAttention forward: {input_shape=}")
        qkv = self.qkv_proj(hidden_states)
        print(f"=======CasualAttention forward after qkv_proj: {qkv.shape=}, {qkv.dtype=}")
        print(f"------ q={qkv[-1, 2028:2048]}, k={qkv[-1, 3052:3072]}, v={qkv[-1, 4076:4096]}  ------")
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
            print(f"=======CasualAttention forward in qk_fuse_norm: {qkv.shape=}, {qkv.dtype=}")
        attn_output = fmha_impl.forward(qkv, kv_cache)
        print(f"=======CasualAttention forward after fmha_impl: {attn_output.shape=}, {attn_output.dtype=}")
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        print(f"=======CasualAttention forward after reshape: {attn_output.shape=}, {attn_output.dtype=}")

        output = self.o_proj(attn_output)
        print(f"=======CasualAttention forward after o_proj: {output.shape=}, {output.dtype=}")
        if self.config.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
