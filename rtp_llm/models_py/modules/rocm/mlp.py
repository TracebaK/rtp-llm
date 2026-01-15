from typing import Dict
import torch
from libth_transformer import rtp_llm_ops
from torch import nn
#import aiter
import torch.nn.functional as F
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import Linear
from rtp_llm.utils.model_weight import W

class DenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        print(f"################## class DenseMLP rocm \n")
        # 拆分gate_proj和up_proj的权重
        ffn13 = weights[W.ffn_w13]
        gate_w, up_w = torch.chunk(ffn13, 2, dim=-1)

        # 拆分gate_proj和up_proj的bias，如果有的话
        ffn13_bias = weights.get(W.ffn_b13, None)
        if ffn13_bias is not None:
            gate_b, up_b = torch.chunk(ffn13_bias, 2, dim=-1)
        else:
            gate_b = None
            up_b = None

        self.gate_proj = Linear(gate_w, gate_b)
        self.up_proj = Linear(up_w, up_b)
        self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))

        if config.activation_type == "SiGLU":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {config.activation_type}")

    def forward(self, x: torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class FusedSiluActDenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        print(f"################ class FusedSiluActDenseMLP rocm \n")
        assert (
            config.activation_type == "SiGLU"
        ), "FusedSiluActDenseMLP only supports SiGLU activation"
        self.gate_up_proj = Linear(weights[W.ffn_w13], weights.get(W.ffn_b13, None))
        self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))

    def _torch_silu_and_mul(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        使用PyTorch实现silu_and_mul功能
        将输入张量分成两半，对前一半应用SiLU激活函数，然后与后一半相乘
        """
        # 确保输入张量的最后一维是偶数
        assert input_tensor.shape[-1] % 2 == 0, "Input tensor last dimension must be even"

        # 将输入张量分成两半
        gate, up = torch.chunk(input_tensor, 2, dim=-1)

        # 应用SiLU激活函数并与另一半相乘
        output_tensor.copy_(nn.functional.silu(gate) * up)

        return output_tensor


    def forward(self, x: torch.Tensor):
        gate_up = self.gate_up_proj(x)

        #d = gate_up.shape[-1] // 2
        #output_shape = gate_up.shape[:-1] + (d,)
        #output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        # aiter.silu_and_mul(output, gate_up)
        #self._torch_silu_and_mul(output, gate_up)
        #down_proj = self.down_proj(output)
        #return down_proj
        gate, up = gate_up.chunk(2, dim=-1)
        output = F.silu(gate) * up
        return self.down_proj(output)


class VllmFusedSiluActDenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        assert (
            config.activation_type == "SiGLU"
        ), "FusedSiluActDenseMLP only supports SiGLU activation"
        from vllm import _custom_ops as ops
        self.gate_up_proj = Linear(weights[W.ffn_w13], weights.get(W.ffn_b13, None))
        self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))
        self.act_fn = ops.silu_and_mul_opt

    def forward(self, x: torch.Tensor):
        x = self.gate_up_proj(x)
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.act_fn(out, x)
        x = self.down_proj(out)
        return x
