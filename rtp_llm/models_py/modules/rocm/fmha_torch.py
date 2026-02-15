import torch
import logging
import time
from typing import Optional, Any, List
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.ops import PyAttentionInputs, FMHAType, KVCache
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from libth_transformer.rtp_llm_ops import FusedRopeKVCachePrefillOp, FusedRopeKVCacheDecodeOp

from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import get_rope
#from transformers.utils import is_flash_attn_2_available
from flash_attn import flash_attn_func, vllm_flash_attn_varlen_func

logger = logging.getLogger(__name__)

class DtkRopeKVCachePrefillOp:
    def __init__(self, gpt_init_parameter: GptInitModelParameters):
        self.gpt_init_parameter = gpt_init_parameter
        self.rotary_emb = get_rope(
            self.gpt_init_parameter.size_per_head, 
            rotary_dim=self.gpt_init_parameter.size_per_head, 
            max_position=40960, 
            base=1000000, 
            rope_scaling=None, 
        )
    def prepare(self, attn_inputs: PyAttentionInputs):
        batch_size = attn_inputs.input_lengths.shape[0]
        block_size = self.gpt_init_parameter.seq_size_per_block

        # 2. 处理KV cache块ID（如果存在）
        kv_cache_block_id_host = None
        kv_cache_block_id_device = None
        
        if attn_inputs.kv_cache_block_id_host.numel() > 0:
            kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host
            kv_cache_block_id_device = attn_inputs.kv_cache_block_id_device
        
        # 3. 计算累积序列长度 (cu_seqlens)
        # 创建CPU上的零张量 [0, 0, 0, ..., 0] 长度为 batch_size + 1
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device='cpu')
        
        # 计算input_lengths的累积和
        # input_lengths.cumsum(0) 会得到 [len1, len1+len2, len1+len2+len3, ...]
        cumulative_lengths = attn_inputs.input_lengths.cumsum(0)
        
        # 将累积长度赋值给cu_seqlens的后半部分
        cu_seqlens[1:] = cumulative_lengths

        # 构造positions
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1] # 计算每个序列的长度 [3, 4]
        positions = torch.ones(cu_seqlens[-1].item(), dtype=torch.long, device='cpu')
        first_indices = cu_seqlens[:-1]
        reset_values = torch.cat([torch.tensor([0]), seq_lens[:-1]])
        positions[first_indices] = 1 - reset_values 
        positions = positions.cumsum(0) - 1
        
        # 构造slot_mapping
        block_indices = positions // block_size
        seq_ids = torch.repeat_interleave(
            torch.arange(len(seq_lens)), 
            seq_lens,
        )
        physical_block_ids = kv_cache_block_id_host[seq_ids, block_indices]
        block_offsets = positions % block_size
        slot_mapping = physical_block_ids * block_size + block_offsets

        # 移动到GPU
        positions = positions.to('cuda')
        slot_mapping = slot_mapping.to('cuda')
        cu_seqlens = cu_seqlens.to('cuda')
        cu_kv_seqlens = cu_seqlens  # KV序列长度通常与Q相同
        
        logging.info(f"DtkRopeKVCachePrefillOp prepare: \n{attn_inputs.kv_cache_block_id_host=}\n{attn_inputs.kv_cache_block_id_device=}\n{attn_inputs.prefix_lengths=}\n{attn_inputs.sequence_lengths=}\n{attn_inputs.input_lengths=}\n{cu_seqlens=}\n{positions=}\n{slot_mapping=}")
        # 4. 准备注意力参数字典
        attn_params = {
            #'attn_type': self._torch_dtype_to_data_type(attn_inputs['dtype']),
            'cu_seqlens': cu_seqlens,
            'cu_kv_seqlens': cu_kv_seqlens,
            'max_seq_len': attn_inputs.input_lengths.max().item(),
            'kv_block_offset': attn_inputs.kv_block_offset,
            'batch_size': batch_size,
            'input_lengths': attn_inputs.input_lengths,
            'kv_cache_block_id_host': kv_cache_block_id_host,
            'kv_cache_block_id_device': kv_cache_block_id_device,
            'kv_cache_dtype': self.gpt_init_parameter.kv_cache_data_type,
            # 添加其他必要的配置
            'head_num': self.gpt_init_parameter.head_num,
            'kv_head_num': self.gpt_init_parameter.head_num_kv,
            'size_per_head': self.gpt_init_parameter.size_per_head,
            'positions': positions,
            'slot_mapping': slot_mapping
        }
        
        return attn_params
    
    def forward(self, qkv: torch.Tensor, fmha_type: FMHAType, kv_cache: Optional[KVCache], params: Optional[Any]) -> torch.Tensor:
        q, k, v = qkv.split([params["head_num"]*params["size_per_head"], 
                             params["kv_head_num"]*params["size_per_head"], 
                             params["kv_head_num"]*params["size_per_head"]], dim=-1)
        q, k = self.rotary_emb(params["positions"], q, k)
        k_reshaped = k.reshape(-1, params["kv_head_num"], params["size_per_head"])
        v_reshaped = v.reshape(-1, params["kv_head_num"], params["size_per_head"])
        k_cache = kv_cache.k_cache_base
        v_cache = kv_cache.v_cache_base
        if kv_cache.k_scale_base is None:
            k_scale = torch.tensor(1.0, device='cuda')
        else:
            k_scale = kv_cache.k_scale_base
        if kv_cache.v_scale_base is None:
            v_scale = torch.tensor(1.0, device='cuda')
        else:
            v_scale = kv_cache.v_scale_base
        logging.info(f"before cache: {k_reshaped.shape=}, {v_reshaped.shape=}, {k_cache.shape=}, {k_cache.dtype=}, {v_cache.shape=}, {v_cache.dtype=}, {params['slot_mapping']=}")
        
        ops.reshape_and_cache_cuda(k_reshaped, 
                                   v_reshaped, 
                                   k_cache,
                                   v_cache,
                                   params["slot_mapping"],
                                   'auto',
                                   k_scale,
                                   v_scale)
        logging.info(f"after kvcache: k={k_cache[1,0,0,:].detach().cpu().tolist()}, v={v_cache[1,0,:,0].detach().cpu().tolist()}")
        return q, k, v


class DtkRopeKVCacheDecodeOp:
    def __init__(self, gpt_init_parameter: GptInitModelParameters):
        self.gpt_init_parameter = gpt_init_parameter
        self.rotary_emb = get_rope(
            self.gpt_init_parameter.size_per_head, 
            rotary_dim=self.gpt_init_parameter.size_per_head, 
            max_position=40960, 
            base=1000000, 
            rope_scaling=None, 
        )

    def prepare(self, attn_inputs: PyAttentionInputs):
        batch_size = attn_inputs.input_lengths.shape[0]
        block_size = self.gpt_init_parameter.seq_size_per_block

        # 2. 处理KV cache块ID（如果存在）
        kv_cache_block_id_host = None
        kv_cache_block_id_device = None
        
        if attn_inputs.kv_cache_block_id_host.numel() > 0:
            kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host
            kv_cache_block_id_device = attn_inputs.kv_cache_block_id_device
        
        # 3. 计算累积序列长度 (cu_seqlens)
        # 创建CPU上的零张量 [0, 0, 0, ..., 0] 长度为 batch_size + 1
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device='cpu')
        
        # 计算input_lengths的累积和
        # input_lengths.cumsum(0) 会得到 [len1, len1+len2, len1+len2+len3, ...]
        cumulative_lengths = attn_inputs.input_lengths.cumsum(0)
        
        # 将累积长度赋值给cu_seqlens的后半部分
        cu_seqlens[1:] = cumulative_lengths

        # 构造positions
        positions = attn_inputs.sequence_lengths.cpu()
        
        # 构造slot_mapping
        seq_lens = attn_inputs.sequence_lengths
        block_indices = positions // block_size
        seq_ids = torch.repeat_interleave(
            torch.arange(len(seq_lens)), 
            1,
        )
        physical_block_ids = kv_cache_block_id_host[seq_ids, block_indices]
        block_offsets = positions % block_size
        slot_mapping = physical_block_ids * block_size + block_offsets

        # 移动到GPU
        positions = positions.long().to('cuda')
        slot_mapping = slot_mapping.long().to('cuda')
        cu_seqlens = cu_seqlens.to('cuda')
        cu_kv_seqlens = cu_seqlens  # KV序列长度通常与Q相同
        
        #logging.info(f"DtkRopeKVCacheDecodeOp prepare: \n{attn_inputs.kv_cache_block_id_host=}\n{attn_inputs.kv_cache_block_id_device=}\n{attn_inputs.prefix_lengths=}\n{attn_inputs.sequence_lengths=}\n{attn_inputs.input_lengths=}\n{cu_seqlens=}\n{positions=}\n{slot_mapping=}")
        # 4. 准备注意力参数字典
        attn_params = {
            #'attn_type': self._torch_dtype_to_data_type(attn_inputs['dtype']),
            'cu_seqlens': cu_seqlens,
            'cu_kv_seqlens': cu_kv_seqlens,
            'max_seq_len': attn_inputs.input_lengths.max().item(),
            'kv_block_offset': attn_inputs.kv_block_offset,
            'batch_size': batch_size,
            'input_lengths': attn_inputs.input_lengths,
            'kv_cache_block_id_host': kv_cache_block_id_host,
            'kv_cache_block_id_device': kv_cache_block_id_device,
            'kv_cache_dtype': self.gpt_init_parameter.kv_cache_data_type,
            # 添加其他必要的配置
            'head_num': self.gpt_init_parameter.head_num,
            'kv_head_num': self.gpt_init_parameter.head_num_kv,
            'size_per_head': self.gpt_init_parameter.size_per_head,
            'positions': positions,
            'slot_mapping': slot_mapping
        }
        
        return attn_params

    def forward(self, qkv: torch.Tensor, fmha_type: FMHAType, kv_cache: Optional[KVCache], params: Optional[Any]) -> torch.Tensor:
        q, k, v = qkv.split([params["head_num"]*params["size_per_head"], 
                             params["kv_head_num"]*params["size_per_head"], 
                             params["kv_head_num"]*params["size_per_head"]], dim=-1)
        #print(f"{params['positions'].shape=}, {q.shape=}, {k.shape=}")
        q, k = self.rotary_emb(params["positions"], q, k)
        k_reshaped = k.reshape(-1, params["kv_head_num"], params["size_per_head"])
        v_reshaped = v.reshape(-1, params["kv_head_num"], params["size_per_head"])
        k_cache = kv_cache.k_cache_base
        v_cache = kv_cache.v_cache_base
        if kv_cache.k_scale_base is None:
            k_scale = torch.tensor(1.0, device='cuda')
        else:
            k_scale = kv_cache.k_scale_base
        if kv_cache.v_scale_base is None:
            v_scale = torch.tensor(1.0, device='cuda')
        else:
            v_scale = kv_cache.v_scale_base
        #logging.info(f"before cache: {q.shape=}, {k.shape=}, {v.shape=}, {k_cache.shape=}, {k_cache.dtype=}, {v_cache.shape=}")
        
        ops.reshape_and_cache_cuda(k_reshaped, 
                                   v_reshaped, 
                                   k_cache,
                                   v_cache,
                                   params["slot_mapping"],
                                   'auto',
                                   k_scale,
                                   v_scale)
        return q

# Simple data structure for fmha_params
class FMHAParams:
    def __init__(self, batch_size: int, max_seq_len: int , seq_lens: Optional[torch.Tensor] = None,
                 kv_cache_block_id_host: Optional[torch.Tensor] = None,
                 kv_cache_block_id_device: Optional[torch.Tensor] = None,
                 input_lengths: Optional[torch.Tensor] = None):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.seq_lens = seq_lens
        self.kv_cache_block_id_host = kv_cache_block_id_host
        self.kv_cache_block_id_device = kv_cache_block_id_device
        self.input_lengths = input_lengths


class TorchNativeFMHAPrefillImplBase(FMHAImplBase):
    def __init__(
        self,
        fmha_impl: Any,
        attn_inputs: PyAttentionInputs,
        config: GptInitModelParameters,
    ) -> None:
        super().__init__(
            fmha_impl,
            DtkRopeKVCachePrefillOp(config.gpt_init_params),
            #FusedRopeKVCachePrefillOp(config.gpt_init_params),
            attn_inputs,
        )

class TorchNativeFMHADecodeImplBase(FMHAImplBase):
    def __init__(
        self,
        fmha_impl: Any,
        attn_inputs: PyAttentionInputs,
        config: GptInitModelParameters,
    ) -> None:
        super().__init__(
            fmha_impl,
            DtkRopeKVCacheDecodeOp(config.gpt_init_params),
            #FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )


PREFILL_MHA_IMPS: List[type[TorchNativeFMHAPrefillImplBase]] = []
DECODE_MHA_IMPS: List[type[TorchNativeFMHADecodeImplBase]] = []



try:
    class TorchNativePrefillImpl(TorchNativeFMHAPrefillImplBase):
        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(TorchNativePrefillAttnOp(config), attn_inputs, config)

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.OPEN_SOURCE


    PREFILL_MHA_IMPS.append(TorchNativePrefillImpl)
except ImportError:
    logging.info("TorchNativePrefillImpl not available, skipped.")


class TorchNativePrefillAttnOp():
    def __init__(
        self, config: GptInitModelParameters
    ):
        self.head_num = config.head_num
        self.head_num_kv = config.head_num_kv
        self.head_dim = config.hidden_size // config.head_num_kv
        self.kv_cache_data_type = config.kv_cache_data_type
        self.softmax_scale = 1 / self.head_dim ** 0.5
    
    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        # 支持所有输入
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # 提取批次大小和最大序列长度
        batch_size = attn_inputs.input_lengths.size(0)
        max_seq_len = attn_inputs.input_lengths.max().item()
        
        # 创建并返回fmha_params对象
        self.fmha_params = FMHAParams(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            input_lengths=attn_inputs.input_lengths,
            kv_cache_block_id_device=attn_inputs.kv_cache_block_id_device,
            kv_cache_block_id_host=attn_inputs.kv_cache_block_id_host
        )
        return self.fmha_params
    
    def forward(self, qkv, kv_cache, fmha_params):
        # logger.info("use dcu flash attentio in prefill")
        # q_tensor: {batch_size, head_num, seq_len, head_dim}
        # k_tensor: {batch_size, head_num_kv, seq_len_with_prefix, head_dim}
        # v_tensor: {batch_size, head_num_kv, seq_len_with_prefix, head_dim}
        q_tensor, k_tensor, v_tensor = qkv[0],qkv[1],qkv[2]
        q_tensor = q_tensor.reshape(-1, self.head_num, self.head_dim)

         # 计算cu_seqlens
        cu_seqlens = torch.zeros(fmha_params.batch_size + 1, dtype=torch.int32, device='cpu')
        # 计算input_lengths的累积和
        # input_lengths.cumsum(0) 会得到 [len1, len1+len2, len1+len2+len3, ...]
        cumulative_lengths = fmha_params.input_lengths.cumsum(0)
        # 将累积长度赋值给cu_seqlens的后半部分
        cu_seqlens[1:] = cumulative_lengths
        cu_seqlens = cu_seqlens.to(q_tensor.device)

        key_cache = kv_cache.k_cache_base
        value_cache = kv_cache.v_cache_base
        output = torch.zeros(q_tensor.shape, dtype=q_tensor.dtype, device=q_tensor.device)
        seqused_k = fmha_params.input_lengths.cuda()

        print(f"{q_tensor.shape=}, {key_cache.shape=}, {value_cache.shape=}")
        vllm_flash_attn_varlen_func(
                    q=q_tensor,
                    k=key_cache,
                    v=value_cache,
                    out=output,
                    cu_seqlens_q=cu_seqlens,
                    max_seqlen_q=cu_seqlens.max().item(),
                    seqused_k=seqused_k,
                    max_seqlen_k=fmha_params.max_seq_len,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                    alibi_slopes=None,
                    window_size=(-1, -1),
                    block_table=fmha_params.kv_cache_block_id_device,
                    softcap=0,
                    scheduler_metadata=None,
                    # fa_version=self.vllm_flash_attn_version,
                    # q_descale=layer._q_scale.expand(descale_shape),
                    # k_descale=layer._k_scale.expand(descale_shape),
                    # v_descale=layer._v_scale.expand(descale_shape),
                    # num_splits=attn_metadata.max_num_splits,
                    is_prefix_cache=True,
        )

        return output

    def forward_fa(self, qkv, kv_cache, fmha_params):
        # logger.info("use dcu flash attentio in prefill")
        # q_tensor: {batch_size, head_num, seq_len, head_dim}
        # k_tensor: {batch_size, head_num_kv, seq_len_with_prefix, head_dim}
        # v_tensor: {batch_size, head_num_kv, seq_len_with_prefix, head_dim}
        q_tensor, k_tensor, v_tensor = qkv[0],qkv[1],qkv[2]
        
        batch_size_actual, head_num_actual, seq_len, head_dim = q_tensor.shape
        
        # dimensions for aiter.flash_attn_func  {batch_size, seq_len, head_num, head_dim}
        q = q_tensor.transpose(1, 2)  # {batch_size, seq_len, head_num, head_dim}
        k = k_tensor.transpose(1, 2)  # {batch_size, seq_len_with_prefix, head_num_kv, head_dim}
        v = v_tensor.transpose(1, 2)  # {batch_size, seq_len_with_prefix, head_dim}

        #res = aiter.flash_attn_func(q, k, v, dropout_p=0., softmax_scale=None, causal=True)
        res = flash_attn_func(q, k, v, dropout_p=0., softmax_scale=None, causal=True)
        
        input_lengths = fmha_params.input_lengths  # 每个 batch 的真实长度
        hidden_size = head_num_actual * head_dim
        
        valid_results = []
        for batch_idx in range(batch_size_actual):
            actual_len = input_lengths[batch_idx].item()
            batch_result = res[batch_idx, :actual_len, :, :]  # {actual_len, head_num, head_dim}
            batch_result = batch_result.reshape(actual_len, hidden_size)  # {actual_len, hidden_size}
            valid_results.append(batch_result)
        
        final_result = torch.cat(valid_results, dim=0)  # {total_token_num, hidden_size}
        return final_result
 
    def forward_torch(self, qkv, kv_cache, fmha_params):
        """
        使用纯PyTorch实现的prefill阶段注意力计算
        """
        logger.debug(f"****TorchNativePrefillAttnOp forward****")

        q_tensor, k_tensor, v_tensor = qkv[0], qkv[1], qkv[2]
        logger.debug(f"{q_tensor.shape=}, {k_tensor.shape=}, {v_tensor.shape=}")

        batch_size, head_num, seq_len, head_dim = q_tensor.shape
        seq_len_with_prefix = k_tensor.shape[2]
        
        # 处理MQA和GQA情况 - 将KV扩展到与Q相同的头数
        if self.head_num != self.head_num_kv:
            # 计算复制因子
            repeat_factor = self.head_num // self.head_num_kv
            
            # 扩展K和V
            k_tensor = k_tensor.repeat_interleave(repeat_factor, dim=1)
            v_tensor = v_tensor.repeat_interleave(repeat_factor, dim=1)
        
        q = q_tensor  # {B, H, L, D}
        k = k_tensor  # {B, H, S, D}
        v = v_tensor  # {B, H, S, D}

        # 计算注意力分数
        scale = 1.0 / (head_dim ** 0.5)

        # 计算Q*K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # {batch_size, seq_len, head_num, seq_len_with_prefix}
        logger.debug(f"after QK {scores.shape=}")
        
        # 遮罩矩阵 - 因果遮罩 (causal mask)
        attn_mask = torch.tril(torch.ones(seq_len, seq_len_with_prefix, device=q.device), diagonal=seq_len_with_prefix - seq_len)
        attn_mask = attn_mask[None, None, :, :]  # {1, 1, seq_len, seq_len_with_prefix}
        logger.debug(f"{attn_mask.shape=}")

        # 应用因果遮罩
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        logger.debug(f"after mask {scores.shape=}")

        # 计算注意力权重
        attn_weights = torch.softmax(scores, dim=-1)  # {batch_size, head_num, seq_len, seq_len_with_prefix}
        
        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, v)  # {batch_size, head_num, seq_len, head_dim}
        logger.debug(f"after v mul {attn_output.shape=}")        
        
        # 转换回原始维度: {batch_size, seq_len, head_num, head_dim}
        attn_output = attn_output.transpose(1, 2).contiguous() # {B, L, H, D}
        logger.debug(f"output reshape {attn_output.shape=}")
        
        # 重塑输出以匹配期望的格式
        input_lengths = fmha_params.input_lengths
        hidden_size = head_num * head_dim
        
        valid_results = []
        for batch_idx in range(batch_size):
            actual_len = input_lengths[batch_idx].item()
            batch_result = attn_output[batch_idx, :actual_len, :, :]  # {head_num, actual_len, head_dim}
            batch_result = batch_result.reshape(actual_len, hidden_size)  # {actual_len, hidden_size}
            valid_results.append(batch_result)
        
        final_result = torch.cat(valid_results, dim=0)  # {total_token_num, hidden_size}
        logger.debug(f"before return {final_result.shape=}")        
        return final_result

try:
    class TorchNativeDecodeImpl(TorchNativeFMHADecodeImplBase):
        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(TorchNativeDecodeAttnOp(config), attn_inputs, config)

    DECODE_MHA_IMPS.append(TorchNativeDecodeImpl)
except ImportError:
    logging.info("TorchNativeDecodeImpl not available, skipped.")
    
class TorchNativeDecodeAttnOp():
    def __init__(
        self, config: GptInitModelParameters
    ):
        self.head_num = config.head_num
        self.head_num_kv = config.head_num_kv
        self.head_dim = config.hidden_size // config.head_num_kv
        self.kv_cache_data_type = config.kv_cache_data_type
        self.softmax_scale = 1 / self.head_dim ** 0.5

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        # 支持所有输入
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # 提取相关参数
        batch_size = attn_inputs.input_lengths.size(0)
        max_seq_len = attn_inputs.input_lengths.max().item()
        seq_lens = attn_inputs.sequence_lengths.cpu() + 1
        seq_lens = seq_lens.cuda()
        kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host
        kv_cache_block_id_device = attn_inputs.kv_cache_block_id_device
        
        # 创建并返回fmha_params对象
        self.fmha_params = FMHAParams(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            seq_lens = seq_lens,
            input_lengths=attn_inputs.input_lengths,
            kv_cache_block_id_host = kv_cache_block_id_host,
            kv_cache_block_id_device = kv_cache_block_id_device
        )
        return self.fmha_params
    
    def forward(self, query: torch.Tensor, kv_cache: Optional[KVCache] , fmha_params:Optional[Any]) -> torch.Tensor:
        q_tensor = query.reshape(-1, self.head_num, self.head_dim)

         # 计算cu_seqlens
        cu_seqlens = torch.zeros(fmha_params.batch_size + 1, dtype=torch.int32, device='cpu')
        # 计算input_lengths的累积和
        # input_lengths.cumsum(0) 会得到 [len1, len1+len2, len1+len2+len3, ...]
        cumulative_lengths = fmha_params.input_lengths.cumsum(0)
        # 将累积长度赋值给cu_seqlens的后半部分
        cu_seqlens[1:] = cumulative_lengths
        cu_seqlens = cu_seqlens.to(q_tensor.device)

        key_cache = kv_cache.k_cache_base
        value_cache = kv_cache.v_cache_base
        output = torch.zeros(q_tensor.shape, dtype=q_tensor.dtype, device=q_tensor.device)
        seqused_k = fmha_params.input_lengths.cuda()

        print(f"{q_tensor.shape=}, {key_cache.shape=}, {value_cache.shape=}, {cu_seqlens=}, {seqused_k=}, k: {key_cache[1, 0, 1, :].detach().cpu().tolist()}, v: {value_cache[1, 0, 1, :].detach().cpu().tolist()}")
        vllm_flash_attn_varlen_func(
                    q=q_tensor,
                    k=key_cache,
                    v=value_cache,
                    out=output,
                    cu_seqlens_q=cu_seqlens,
                    max_seqlen_q=cu_seqlens.max().item(),
                    seqused_k=seqused_k,
                    max_seqlen_k=fmha_params.max_seq_len,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                    alibi_slopes=None,
                    window_size=(-1, -1),
                    block_table=fmha_params.kv_cache_block_id_device,
                    softcap=0,
                    scheduler_metadata=None,
                    # fa_version=self.vllm_flash_attn_version,
                    is_prefix_cache=True,
        )

        return output

    def forward_fa_aiter_layout(self, query: torch.Tensor, kv_cache: Optional[KVCache] , fmha_params:Optional[Any]) -> torch.Tensor:
        # 1. 基础参数准备
        num_seqs, num_heads, head_size = query.shape # [batch_size, q_num_heads, head_size]
        seq_lens = fmha_params.seq_lens
        max_seqlen_k = seq_lens.max().item()
        # max_seqlen_k = fmha_params.max_seq_len + 1
    
        # 2. 获取 KV Cache 引用
        # 你的维度: [num_blocks, num_kv_heads, block_size, head_size]
        k_cache = kv_cache.k_cache_base
        # logger.info(f"原始vcache: {kv_cache.v_cache_base[1,0,0,:]}")
        v_cache = kv_cache.v_cache_base.permute(0, 1, 3, 2)

        # ======================= aiter layout to vllm layout =======================
        num_blocks, num_kv_heads, block_size, _ = k_cache.shape
        x = 8
        k_cache = k_cache.view(
            num_blocks, 
            num_kv_heads, 
            head_size // x,    # 16
            block_size // 16,  # 4
            16,                # 每个 tile 的 token 数
            x                  # 8
        )

        # 3. 重新排列维度
        # 目标顺序: [block, head, (block_size//16, 16), (head_size//x, x)]
        # 对应索引: [0, 1, (3, 4), (2, 5)]
        k_cache = k_cache.permute(0, 1, 3, 4, 2, 5).contiguous()

        # 4. 合并维度得到最终形状 [num_blocks, num_heads, 64, 128]
        k_cache = k_cache.view(num_blocks, num_kv_heads, block_size, head_size)
        
        v_cache = v_cache.view(num_blocks, num_kv_heads, 1, head_size, block_size)
        v_cache = v_cache.permute(0, 1, 2, 4, 3).contiguous()
        v_cache = v_cache.view(num_blocks, num_kv_heads, head_size, block_size)
        # ===================== aiter layout to vllm layout =========================

        # 3. 构造 varlen 算子需要的 metadata
        # Decode 阶段，每个 sequence 的 q 长度为 1
        device = query.device
        cu_seqlens_q = torch.arange(0, num_seqs + 1, device=device, dtype=torch.int32)
        max_seqlen_q = 1
    
        # 计算缩放系数
        softmax_scale = 1.0 / (head_size ** 0.5)

        # 4. FP8 缩放处理 (如果有)
        # 注意：vllm 接口通常使用 descale (1/scale)
        # k_descale = 1.0 / kv_cache.k_scale_base if hasattr(kv_cache, 'k_scale_base') else None
        # v_descale = 1.0 / kv_cache.v_scale_base if hasattr(kv_cache, 'v_scale_base') else None

        # 5. 调用新接口
        # 注意：我们直接传入 k_cache，如果算子支持特定的 stride，则不需要显式 reshape
        # 如果接口强制要求特定 layout，这里使用 view/permute（在 PyTorch 中通常只是产生 view，不触发 copy）
        # logger.info(f"{query.shape=}, {k_cache.shape=}, {v_cache.shape=}, {max_seqlen_q=}, cu_seqlens_q={cu_seqlens_q.detach().cpu().tolist()}, seqused_k={seq_lens.detach().cpu().tolist()}, {max_seqlen_k=}, block_table={fmha_params.kv_cache_block_id_device.detach().cpu().tolist()}, kcache[1,0,0,:]: {k_cache[1,0,0,:].detach().cpu().tolist()}, vcache[1,0,:,0]: {v_cache[1,0,:,0].detach().cpu().tolist()}")
        output = vllm_flash_attn_varlen_func(
            q=query,
            k=k_cache, # 内部会根据 block_table 索引
            v=v_cache,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            seqused_k=seq_lens,
            softmax_scale=softmax_scale,
            causal=True, # Decode 阶段通常视为 causal
            block_table=fmha_params.kv_cache_block_id_device,
            kv_cache_dtype="auto", # e.g., "fp8" 或 "auto"
            k_descale=None,
            v_descale=None,
        )
        # logger.info(f"{output.shape=}, 输出最后一维的前20个数：{output[...,-1][:20].detach().cpu().tolist()}")

        # 6. 还原输出形状
        return output.view(num_seqs, -1) 
 
    def forward_pa(self, query: torch.Tensor, kv_cache: Optional[KVCache] , fmha_params:Optional[Any]) -> torch.Tensor:
        logger.info(f"query shape in decode forward: {query.shape}")
        seq_lens = fmha_params.seq_lens
        max_seq_len = fmha_params.max_seq_len + 1
        key_cache = kv_cache.k_cache_base
        value_cache = kv_cache.v_cache_base
        block_tables_id_host = fmha_params.kv_cache_block_id_host
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        num_kv_heads = self.head_num_kv
        scale = 1.0 / (self.head_dim ** 0.5)
        alibi_slopes = None

        k_scale = kv_cache.k_scale_base if kv_cache and kv_cache.k_scale_base is not None else 1.0
        v_scale = kv_cache.v_scale_base if kv_cache and kv_cache.v_scale_base is not None else 1.0
        # 将k_scale和v_scale转换为tensor类型，如果它们不是tensor的话
        if not isinstance(k_scale, torch.Tensor):
            k_scale = torch.tensor(k_scale, dtype=query.dtype, device=query.device)
        if not isinstance(v_scale, torch.Tensor):
            v_scale = torch.tensor(v_scale, dtype=query.dtype, device=query.device)

        max_num_blocks = block_tables_id_device.shape[1]

        num_seqs, num_heads, head_size = query.shape
        block_size = value_cache.shape[2]
        _PARTITION_SIZE_ROCM = 256
        
        # init output
        output = torch.empty_like(query)

        max_num_partitions = (max_seq_len + _PARTITION_SIZE_ROCM - 1) // _PARTITION_SIZE_ROCM
        assert _PARTITION_SIZE_ROCM % block_size == 0
        # init tmp_output
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        
        # init exp_sums 
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        fp8_out_scale=None
        cpa_fp8_out = False
        # init max_logits 
        max_logits = torch.ones_like(exp_sums)
        
        kv_cache_dtype ="auto"
        key_cache_reshaped = key_cache.permute(0,1,3,2)
        value_cache_reshaped = value_cache.permute(0,1,3,2)

        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache_reshaped,
            value_cache_reshaped,
            num_kv_heads,
            float(scale),
            block_tables_id_device,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,  # kv_cache_dtype
            k_scale,
            v_scale,
        )

        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped

    def forward_torch(self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params: Optional[Any]) -> torch.Tensor:
        """
        使用纯PyTorch实现的decode阶段注意力计算
        """
        # 获取参数
        seq_lens = fmha_params.seq_lens
        key_cache = kv_cache.k_cache_base
        value_cache = kv_cache.v_cache_base
        logger.debug(f"==========TorchNativeDecodeAttnOp forward: {query.shape=}, {key_cache.shape=}, {value_cache.shape=}") 
        logger.debug(f"=========={torch.nonzero(key_cache[:, -1, :, -1]).cpu().tolist()=}")

        # 辅助变量
        block_tables = fmha_params.kv_cache_block_id_device
        num_kv_heads = key_cache.shape[1]
        num_q_heads = query.shape[1]
        head_dim = query.shape[2]
        block_size = value_cache.shape[2]
        scale = 1.0 / (self.head_dim ** 0.5)

        # init output
        output = torch.empty_like(query)
        num_seqs = query.shape[0]
        
        logger.debug(f"=========={seq_lens=}")
        # PyTorch实现的paged attention替代方案
        for seq_idx in range(num_seqs):
            # 获取当前序列的序列长度
            cur_seq_len = seq_lens[seq_idx].item()

            # 获取当前序列的块表
            block_table = block_tables[seq_idx]
            logger.debug(f"=========={block_table=}")

            # 计算需要访问的块数量
            num_blocks = (cur_seq_len + block_size - 1) // block_size
            logger.debug(f"=========={num_blocks=}")

            # 提取块ID并获取数据
            block_ids = block_table[:num_blocks] # [num_blocks]

            # 使用高级索引一次性取出所有块，避免在循环中逐个取
            # k_blocks shape: [num_blocks, kv_head_num, block_size, head_dim]
            k_blocks = key_cache[block_ids] 
            v_blocks = value_cache[block_ids]

            # 2. 变换维度以进行拼接
            # 原状: [num_blocks, kv_head, block_size, dim]
            # 目标: [num_blocks, block_size, kv_head, dim] -> 展平 -> [total_len, kv_head, dim]
            # permute(0, 2, 1, 3): 交换 block_size 和 kv_head
            k_blocks = k_blocks.permute(0, 2, 1, 3) 
            v_blocks = v_blocks.permute(0, 2, 1, 3)

            # 展平前两维 (Blocks * BlockSize) -> Time
            # shape: [num_blocks * block_size, kv_head, dim]
            keys = k_blocks.reshape(-1, num_kv_heads, head_dim)
            values = v_blocks.reshape(-1, num_kv_heads, head_dim)
            
            # 截断 Padding (去掉最后一个块中多余的部分)
            keys = keys[:cur_seq_len]
            values = values[:cur_seq_len]


            logger.debug(f"=========={keys.shape=}, {keys[..., -1].detach().cpu().to(torch.float32).tolist()}")

            # 3. 处理 GQA/MQA (扩展KV头数)
            if num_q_heads != num_kv_heads:
                repeat = num_q_heads // num_kv_heads
                keys = keys.repeat_interleave(repeat, dim=1)   # [seq_len, q_head, dim]
                values = values.repeat_interleave(repeat, dim=1)
            logger.debug(f"==========Pytorch paged attention {keys.shape=}, {values.shape=}")
        
            # 4. Attention 计算
            # q: [1, H, D]
            q = query[seq_idx:seq_idx+1] 
            
            # 计算 Score = Q * K^T
            # q: [1, H, 1, D]
            # k: [seq_len, H, D] -> permute -> [1, H, D, seq_len]
            # score: [1, H, 1, seq_len]
            scores = torch.matmul(q.unsqueeze(2), keys.permute(1, 2, 0).unsqueeze(0)) * scale
            
            # Softmax (Decode阶段不需要 Mask，因为可以看到所有历史)
            attn_weights = torch.softmax(scores, dim=-1)
            
            # 计算 Output = Score * V
            # weight: [1, H, 1, seq_len]
            # v: [seq_len, H, D] -> permute -> [1, H, seq_len, D]
            # out: [1, H, 1, D]
            attn_out = torch.matmul(attn_weights, values.permute(1, 0, 2).unsqueeze(0))
            
            output[seq_idx] = attn_out.squeeze(2).squeeze(0)

        return output.reshape(num_seqs, -1)
