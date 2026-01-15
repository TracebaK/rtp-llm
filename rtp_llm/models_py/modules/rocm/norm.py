from typing import Tuple, Union
import logging

import torch
import torch.nn.functional as F
#from aiter import layernorm2d_fwd as layernorm2d_fwd
#from aiter import rmsnorm2d_fwd as rms_norm
from libth_transformer import rtp_llm_ops
from torch import nn
from lightop import op
from vllm import _custom_ops as ops


from rtp_llm.models_py.modules.norm import BaseNorm


logger = logging.getLogger(__name__)


class BaseLayerNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class LayerNormTorch(BaseLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(dim=-1, keepdim=True)
        squared_sum = (hidden_states**2).mean(dim=-1, keepdim=True)

        x_normalized = (hidden_states - mean) / torch.sqrt(
            (squared_sum - (mean**2)) + self.variance_epsilon
        )
        return (self.weight * x_normalized + self.beta).to(input_dtype)


class LayerNorm(BaseLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(self, hidden_states: torch.Tensor):
        output = torch.empty_like(hidden_states)
        rtp_llm_ops.layernorm(
            output, hidden_states, self.weight.data, self.beta, self.variance_epsilon, 0
        )
        return output


class LightopRMSNorm(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states):
        logger.info(f"LightopRMSNorm {hidden_states.is_contiguous()=}, {hidden_states.shape}, {hidden_states.dtype}")
        return op.rmsnorm_forward_autograd(hidden_states, self.weight, self.variance_epsilon, False)

class VllmRMSNorm(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logger.debug(f"VllmRMSNorm {hidden_states.is_contiguous()=}, {hidden_states.shape}, {hidden_states.dtype}")
        out = torch.empty_like(hidden_states)
        if True:
            ops.rms_norm_opt(out, hidden_states, self.weight, self.variance_epsilon)
        else:
            ops.rms_norm(out, x, self.weight, self.variance_epsilon)
        return out
    
class LightopRMSNormAdd(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states, residual):
        logger.info(f"LightopRMSNormAdd {hidden_states.is_contiguous()=}, {hidden_states.shape}, {hidden_states.dtype}")
        return op.rn_add_forward_autograd(hidden_states, residual, self.weight, self.variance_epsilon, False, False)


class RMSNorm(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states: torch.Tensor):
        logger.debug(f"RMSNorm {hidden_states.shape}, {hidden_states.dtype}")
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight is not None:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states * self.weight.to(torch.float32)
                hidden_states = hidden_states.to(self.weight.dtype)
            else:
                hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states


class BaseAddBiasResLayerNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()


class AddBiasResLayerNormROCmTorch(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor
    ):
        output = F.layer_norm(
            input=hidden_states,
            normalized_shape=(hidden_states.shape[-1],),
            weight=self.weight.data,
            bias=bias,
            eps=self.variance_epsilon,
        )
        return output


class AddBiasResLayerNorm(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        bias: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if hidden_states.shape[0] > 32 and hidden_states.shape[1] <= 768:
            #return layernorm2d_fwd(
            return op.layernorm_forward_autograd(
                hidden_states,
                self.weight,
                bias,
                self.variance_epsilon,
                False,
            )
        else:
            rtp_llm_ops.fused_add_layernorm(
                hidden_states,
                residual,
                bias,
                self.weight.data,
                self.beta,
                self.variance_epsilon,
                0,
            )
            return hidden_states


class AddBiasResLayerNormTorch(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor
    ):
        hidden_states = hidden_states + bias + residual
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(dim=-1, keepdim=True)
        squared_sum = (hidden_states**2).mean(dim=-1, keepdim=True)

        x_normalized = (hidden_states - mean) / torch.sqrt(
            (squared_sum - (mean**2)) + self.variance_epsilon
        )
        return (self.weight * x_normalized + self.beta).to(input_dtype)


class QKRMSNorm(nn.Module):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: float = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.q_norm = VllmRMSNorm(q_weight, eps)
        self.k_norm = VllmRMSNorm(k_weight, eps)
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.q_size = self.head_num * self.size_per_head
        self.kv_size = self.kv_head_num * self.size_per_head
        self.variance_epsilon = eps

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.size_per_head)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.size_per_head)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(self, hidden_states):
        q, k, v = hidden_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        output = torch.cat([q, k, v], dim=-1)
        return output


class FusedQKRMSNorm(nn.Module):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: float = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.eps = eps
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.q_size = self.head_num * self.size_per_head
        self.kv_size = self.kv_head_num * self.size_per_head

    def forward(self, hidden_states):
        logger.debug(f"FusedQKRMSNorm forward: {hidden_states.shape=}, {hidden_states.dtype=}")
        # 保存原始数据类型
        input_dtype = hidden_states.dtype
        
        # 分离Q、K、V部分
        q, k, v = hidden_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        logger.debug(f"FusedQKRMSNorm forward: {q.is_contiguous()=}, {k.is_contiguous()=}, {v.is_contiguous()=}") 
        # 对Q部分应用RMSNorm
        q = q.reshape(-1, self.size_per_head)
        q = op.rmsnorm_forward_autograd(q, self.q_weight, self.eps, False)
        q = q.view(-1, self.q_size)

        # 对K部分应用RMSNorm
        k = k.reshape(-1, self.size_per_head)
        k = op.rmsnorm_forward_autograd(k, self.k_weight, self.eps, False) 
        k = k.view(-1, self.kv_size)

        # 合并结果
        output = torch.cat([q, k, v], dim=-1)
        logger.debug(f"FusedQKRMSNorm forward: output {output.shape=}")
        return output
