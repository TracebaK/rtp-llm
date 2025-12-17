import torch
import logging
from typing import Optional, Any, List
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.ops import PyAttentionInputs, FMHAType, KVCache
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from libth_transformer.rtp_llm_ops import FusedRopeKVCachePrefillOp, FusedRopeKVCacheDecodeOp

from vllm import _custom_ops as ops
#from transformers.utils import is_flash_attn_2_available
#from flash_attn import flash_attn_func

import os
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


class TorchNativeRopeKVCachePrefillOp:
    def __init__(self, gpt_init_params):
        self.gpt_init_params = gpt_init_params
        self.rope_config = getattr(gpt_init_params, 'rope_config', None)
        
    def prepare(self, attn_inputs):
        batch_size = attn_inputs.input_lengths.size(0)
        max_seq_len = attn_inputs.input_lengths.max().item()
        
        # Create and return fmha_params with the required attributes
        self.fmha_params = FMHAParams(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            input_lengths=attn_inputs.input_lengths
        )
        return self.fmha_params
    
    def forward(self, qkv, fmha_type, kv_cache, params):
        print("TorchNativeRopeKVCachePrefillOp.forward called")
        # Extract Q, K, V from qkv
        # qkv is expected to be of shape [token_num, head_num * 3 * size_per_head]
        batch_size = params.batch_size
        max_seq_len = params.max_seq_len
        
        # Reshape qkv to separate Q, K, V
        print(f"TorchNativeRopeKVCachePrefillOp {self.gpt_init_params=}")
        head_num = self.gpt_init_params.head_num if hasattr(self.gpt_init_params, 'head_num') else 32
        head_num_kv = getattr(self.gpt_init_params, 'head_num_kv', head_num)
        size_per_head = getattr(self.gpt_init_params, 'size_per_head', 128)
        
        # Reshape to separate Q, K, V
        qkv_reshaped = qkv.view(-1, head_num + 2 * head_num_kv, size_per_head)
        token_num = qkv_reshaped.size(0)
        
        # Separate Q, K, V
        q = qkv_reshaped[:, :head_num, :]  # [token_num, head_num, size_per_head]
        k = qkv_reshaped[:, head_num:head_num + head_num_kv, :]  # [token_num, head_num_kv, size_per_head]
        v = qkv_reshaped[:, head_num + head_num_kv:, :]  # [token_num, head_num_kv, size_per_head]
        
        # Reshape to 4D tensors [batch_size, head_num, seq_len, size_per_head]
        # We need to split tokens according to input_lengths
        input_lengths = params.input_lengths
        cu_seqlens = torch.cat([torch.tensor([0]), input_lengths.cumsum(0)])
        
        # Create output tensors
        q_output = torch.zeros(batch_size, head_num, max_seq_len, size_per_head, 
                              dtype=q.dtype, device=q.device)
        k_output = torch.zeros(batch_size, head_num_kv, max_seq_len, size_per_head, 
                              dtype=k.dtype, device=k.device)
        v_output = torch.zeros(batch_size, head_num_kv, max_seq_len, size_per_head, 
                              dtype=v.dtype, device=v.device)
        
        # Fill tensors based on cu_seqlens
        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i+1]
            seq_len = end_idx - start_idx
            
            # Copy Q, K, V for this batch
            q_output[i, :, :seq_len, :] = q[start_idx:end_idx].transpose(0, 1)
            k_output[i, :, :seq_len, :] = k[start_idx:end_idx].transpose(0, 1)
            v_output[i, :, :seq_len, :] = v[start_idx:end_idx].transpose(0, 1)
            
        # Apply RoPE if configured
        if self.rope_config is not None:
            q_output, k_output = self._apply_rope(q_output, k_output, max_seq_len)
        
        # Store to KV cache if needed
        if kv_cache is not None:
            self._store_kv_cache(k_output, v_output, kv_cache, params)
        
        return (q_output, k_output, v_output)
    
    def _apply_rope(self, q, k, max_seq_len):
        """Apply rotary position embedding to Q and K tensors"""
        batch_size, head_num, seq_len, head_dim = q.shape
        _, head_num_kv, _, _ = k.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        
        # Create cos/sin caches
        cos, sin = self._create_cos_sin_cache(seq_len, head_dim, device=q.device)
        
        # Apply RoPE
        q_embed = self._rotate_qk(q, cos, sin, position_ids)
        k_embed = self._rotate_qk(k, cos, sin, position_ids)
        
        return q_embed, k_embed
    
    def _create_cos_sin_cache(self, seq_len, head_dim, device='cuda', base=10000):
        """Creates cos and sin caches for rotary positional embeddings"""
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        freqs = torch.outer(position, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cache = emb.cos()
        sin_cache = emb.sin()
        return cos_cache, sin_cache
    
    def _rotate_qk(self, x, cos, sin, position_ids):
        """Rotate query/key tensors with cos/sin caches"""
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Handle position_ids
        cos = cos[position_ids].unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
        sin = sin[position_ids].unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
        
        # Split tensor into two halves
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        
        # Apply rotation
        rotated = torch.cat((-x2, x1), dim=-1)
        
        # Apply RoPE transformation
        x_out = (x * cos) + (rotated * sin)
        return x_out
    
    def _store_kv_cache(self, k, v, kv_cache, params):
        """Store K and V to KV cache"""
        # In a full implementation, this would handle storing to the paged KV cache
        # For now, we'll leave this as a placeholder
        pass

class TorchNativeRopeKVCacheDecodeOp:
    def __init__(self, gpt_init_params):
        self.gpt_init_params = gpt_init_params
        self.rope_config = getattr(gpt_init_params, 'rope_config', None)
        
    def prepare(self, attn_inputs):
        batch_size = attn_inputs.sequence_lengths.size(0)
        max_seq_len = attn_inputs.input_lengths.max().item()
        seq_lens = attn_inputs.sequence_lengths.cpu() + 1
        seq_lens = seq_lens.cuda()
        kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host
        kv_cache_block_id_device = attn_inputs.kv_cache_block_id_device
        
        # Create and return fmha_params with the required attributes
        self.fmha_params = FMHAParams(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            kv_cache_block_id_host=kv_cache_block_id_host,
            kv_cache_block_id_device=kv_cache_block_id_device,
            input_lengths=attn_inputs.input_lengths
        )
        return self.fmha_params
    
    def forward(self, qkv, fmha_type, kv_cache, params):
        # For decode, qkv contains only the new token's QKV
        head_num = self.gpt_init_params.head_num if hasattr(self.gpt_init_params, 'head_num') else 32
        head_num_kv = getattr(self.gpt_init_params, 'head_num_kv', head_num)
        size_per_head = getattr(self.gpt_init_params, 'size_per_head', 128)
        
        # Reshape qkv to separate Q, K, V
        qkv_reshaped = qkv.view(-1, head_num + 2 * head_num_kv, size_per_head)
        token_num = qkv_reshaped.size(0)
        
        # Separate Q, K, V
        q = qkv_reshaped[:, :head_num, :]  # [token_num, head_num, size_per_head]
        k = qkv_reshaped[:, head_num:head_num + head_num_kv, :]  # [token_num, head_num_kv, size_per_head]
        v = qkv_reshaped[:, head_num + head_num_kv:, :]  # [token_num, head_num_kv, size_per_head]
        
        # For decode, we typically have token_num == batch_size
        batch_size = token_num
        
        # Reshape to 3D tensors [batch_size, head_num, size_per_head]
        q_output = q.view(batch_size, head_num, size_per_head)
        
        # Apply RoPE to query if configured
        if self.rope_config is not None:
            q_output = self._apply_rope_decode(q_output, params.seq_lens)
        
        # Store K,V to cache if needed
        if kv_cache is not None:
            self._store_kv_cache_decode(k, v, kv_cache, params)
        
        return q_output
    
    def _apply_rope_decode(self, q, seq_lens):
        """Apply rotary position embedding to Q tensor during decode"""
        batch_size, head_num, head_dim = q.shape
        
        # Get current positions
        position_ids = (seq_lens - 1).unsqueeze(1)  # [batch_size, 1]
        
        # Create cos/sin caches
        max_pos = seq_lens.max().item()
        cos, sin = self._create_cos_sin_cache(max_pos, head_dim, device=q.device)
        
        # Apply RoPE
        cos = cos[position_ids].unsqueeze(2)  # [batch_size, 1, 1, head_dim]
        sin = sin[position_ids].unsqueeze(2)  # [batch_size, 1, 1, head_dim]
        
        # Split tensor into two halves
        q1 = q[..., :head_dim // 2]
        q2 = q[..., head_dim // 2:]
        
        # Apply rotation
        rotated = torch.cat((-q2, q1), dim=-1)
        
        # Apply RoPE transformation
        q_out = (q * cos.squeeze(1)) + (rotated * sin.squeeze(1))
        return q_out
    
    def _create_cos_sin_cache(self, seq_len, head_dim, device='cuda', base=10000):
        """Creates cos and sin caches for rotary positional embeddings"""
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        freqs = torch.outer(position, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cache = emb.cos()
        sin_cache = emb.sin()
        return cos_cache, sin_cache
    
    def _store_kv_cache_decode(self, k, v, kv_cache, params):
        """Store K and V to KV cache during decode"""
        # In a full implementation, this would handle storing to the paged KV cache
        # For now, we'll leave this as a placeholder
        pass

class TorchNativeFMHAPrefillImplBase(FMHAImplBase):
    def __init__(
        self,
        fmha_impl: Any,
        attn_inputs: PyAttentionInputs,
        config: GptInitModelParameters,
    ) -> None:
        super().__init__(
            fmha_impl,
            FusedRopeKVCachePrefillOp(config.gpt_init_params),
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
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
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
        self.head_dim = config.hidden_size // config.head_num
        self.head_num_kv = config.head_num_kv
        self.kv_cache_data_type = config.kv_cache_data_type

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
            input_lengths=attn_inputs.input_lengths
        )
        return self.fmha_params

    def forward(self, qkv, kv_cache, fmha_params):
        """
        使用纯PyTorch实现的prefill阶段注意力计算
        """
        print(f"======TorchNativePrefillAttnOp forward, {fmha_params=}")
        # q_tensor: {batch_size, head_num, seq_len, head_dim}
        # k_tensor: {batch_size, head_num_kv, seq_len_with_prefix, head_dim}
        # v_tensor: {batch_size, head_num_kv, seq_len_with_prefix, head_dim}
        q_tensor, k_tensor, v_tensor = qkv[0], qkv[1], qkv[2]
        print(f"======TorchNativePrefillAttnOp {q_tensor.shape=}, {k_tensor.shape=}, {v_tensor.shape=}")
        print(f"======TorchNativePrefillAttnOp {q_tensor.dtype=}, {k_tensor.dtype=}, {v_tensor.dtype=}")

        batch_size, head_num, seq_len, head_dim = q_tensor.shape
        seq_len_with_prefix = k_tensor.shape[2]
        
        # 处理MQA和GQA情况 - 将KV扩展到与Q相同的头数
        if self.head_num != self.head_num_kv:
            # 计算复制因子
            repeat_factor = self.head_num // self.head_num_kv
            
            # 扩展K和V
            k_tensor = k_tensor.repeat_interleave(repeat_factor, dim=1)
            v_tensor = v_tensor.repeat_interleave(repeat_factor, dim=1)
        
        # 转换维度以适应注意力计算: {batch_size, seq_len, heads, head_dim}
        q = q_tensor.transpose(1, 2).reshape(batch_size, seq_len, -1)  # {batch_size, seq_len, head_num*head_dim}
        k = k_tensor.transpose(1, 2).reshape(batch_size, seq_len, -1)  # {batch_size, seq_len_with_prefix, head_num*head_dim}
        v = v_tensor.transpose(1, 2).reshape(batch_size, seq_len, -1)  # {batch_size, seq_len_with_prefix, head_num*head_dim}
        print(f"======TorchNativePrefillAttnOp after transpose {q.shape=}, {k.shape=}, {v.shape=}")

        # 计算注意力分数
        scale = 1.0 / (head_dim ** 0.5)
        
        # 遮罩矩阵 - 因果遮罩 (causal mask)
        attn_mask = torch.tril(torch.ones(seq_len, seq_len_with_prefix, device=q.device), diagonal=seq_len_with_prefix - seq_len)
        attn_mask = attn_mask.unsqueeze(0)  # {1, seq_len, seq_len_with_prefix}
        print(f"======TorchNativePrefillAttnOp {attn_mask.shape=}")

        # 计算Q*K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # {batch_size, seq_len, head_num, seq_len_with_prefix}
        print(f"======TorchNativePrefillAttnOp after QK {scores.shape=}")

        # 应用因果遮罩
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        print(f"======TorchNativePrefillAttnOp after mask {scores.shape=}")

        # 计算注意力权重
        attn_weights = torch.softmax(scores, dim=-1)  # {batch_size, seq_len, head_num, seq_len_with_prefix}
        
        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, v)  # {batch_size, seq_len, head_num*head_dim}
        print(f"======TorchNativePrefillAttnOp after v mul {attn_output.shape=}")        
        
        # 转换回原始维度: {batch_size, head_num, seq_len, head_dim}
        attn_output = attn_output.reshape(batch_size, seq_len, head_num, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 重塑输出以匹配期望的格式
        input_lengths = fmha_params.input_lengths
        hidden_size = head_num * head_dim
        
        valid_results = []
        for batch_idx in range(batch_size):
            actual_len = input_lengths[batch_idx].item()
            batch_result = attn_output[batch_idx, :, :actual_len, :]  # {head_num, actual_len, head_dim}
            batch_result = batch_result.transpose(0, 1).reshape(actual_len, hidden_size)  # {actual_len, hidden_size}
            valid_results.append(batch_result)
        
        final_result = torch.cat(valid_results, dim=0)  # {total_token_num, hidden_size}
        print(f"======TorchNativePrefillAttnOp before return {final_result.shape=}")        
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
        self.head_dim = config.hidden_size // config.head_num
        self.head_num_kv = config.head_num_kv
        self.kv_cache_data_type = config.kv_cache_data_type

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
            kv_cache_block_id_host = kv_cache_block_id_host,
            kv_cache_block_id_device = kv_cache_block_id_device
        )
        return self.fmha_params

    def forward(self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params: Optional[Any]) -> torch.Tensor:
        """
        使用纯PyTorch实现的decode阶段注意力计算
        """
        # 获取参数
        seq_lens = fmha_params.seq_lens
        max_seq_len = fmha_params.max_seq_len + 1
        key_cache = kv_cache.k_cache_base if kv_cache else None
        value_cache = kv_cache.v_cache_base if kv_cache else None
        print(f"======TorchNativeDecodeAttnOp forward: {key_cache.shape=}, {value_cache.shape=}") 
        print(f"======TorchNativeDecodeAttnOp forward: {query.shape=}, {query.dtype=}")

        block_tables_id_host = fmha_params.kv_cache_block_id_host
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        num_kv_heads = self.head_num_kv
        scale = 1.0 / (self.head_dim ** 0.5)
        alibi_slopes = None

        k_scale = kv_cache.k_scale_base if kv_cache and kv_cache.k_scale_base is not None else 1.0
        v_scale = kv_cache.v_scale_base if kv_cache and kv_cache.v_scale_base is not None else 1.0
        max_num_blocks = block_tables_id_device.shape[1]

        num_seqs, num_heads, head_size = query.shape
        block_size = value_cache.shape[2]
        _PARTITION_SIZE_ROCM = 256

        # init output
        output = torch.empty_like(query)
        print("======{output.shape=}")

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
        
        print("======Pytorch paged attention start")
        # PyTorch实现的paged attention替代方案
        for seq_idx in range(num_seqs):
            # 获取当前序列的序列长度
            cur_seq_len = seq_lens[seq_idx].item()

            # 获取当前序列的块表
            block_table = block_tables_id_device[seq_idx]
            print(f"======{block_table=}")

            # 计算需要访问的块数量
            num_blocks = (cur_seq_len + block_size - 1) // block_size
            print(f"======{num_blocks=}")

            # 收集所有相关的key和value
            all_keys = []
            all_values = []

            for block_idx in range(num_blocks):
                block_id = block_table[block_idx].item()

                # 从缓存中提取key和value块
                # 原始形状: [num_blocks, num_kv_heads, block_size, head_size]
                k_block = key_cache[block_id]   # [num_kv_heads, block_size, head_size]
                v_block = value_cache[block_id] # [num_kv_heads, block_size, head_size]
               
                # 调整维度顺序为[block_size, num_kv_heads, head_size]
                k_block = k_block.permute(1, 0, 2)
                v_block = v_block.permute(1, 0, 2)
                print(f"======{k_block.shape=}, {v_block.shape=}")

                all_keys.append(k_block)
                all_values.append(v_block)

            # 合并所有块
            if all_keys:
                keys = torch.cat(all_keys, dim=0)[:cur_seq_len]  # [cur_seq_len, num_kv_heads, head_size]
                values = torch.cat(all_values, dim=0)[:cur_seq_len]  # [cur_seq_len, num_kv_heads, head_size]

                # 处理MQA/GQA情况 - 如果需要，扩展KV头数以匹配Q头数
                if num_heads != num_kv_heads:
                    repeat_factor = num_heads // num_kv_heads
                    keys = keys.repeat_interleave(repeat_factor, dim=1)    # [cur_seq_len, num_heads, head_size]
                    values = values.repeat_interleave(repeat_factor, dim=1)  # [cur_seq_len, num_heads, head_size]
                    print(f"======In GQA: {keys.shape=}, {values.shape=}")

                # 获取当前查询向量
                q = query[seq_idx:seq_idx+1]  # [1, num_heads, head_size]
                print(f"======{q.shape=}")

                # 计算注意力分数: Q * K^T
                # q: [1, num_heads, head_size]
                # keys: [cur_seq_len, num_heads, head_size]
                scores = torch.matmul(q.unsqueeze(2), keys.permute(1, 2, 0).unsqueeze(0))  # [1, num_heads, 1, cur_seq_len]
                scores = scores * scale  # [1, num_heads, cur_seq_len]
                print(f"=====before mask {scores.shape=}")

                # 计算注意力权重
                triu_mask = torch.triu(torch.full((cur_seq_len, cur_seq_len), True, device=q.device), diagonal=1)[-1:]          # [1, seq_len]
                scores = scores.masked_fill(triu_mask, float('-inf'))
                print(f"=====after mask {scores.shape=}")
                attn_weights = torch.softmax(scores, dim=-1)  # [1, num_heads, cur_seq_len]
                print(f"====={attn_weights.shape=}")

                # 应用注意力权重到values
                # attn_weights: [1, num_heads, 1, cur_seq_len]
                # values: [cur_seq_len, num_heads, head_size]
                result = torch.matmul(attn_weights, values.permute(1, 0, 2).unsqueeze(0)).squeeze(2)  # [1, num_heads, head_size]
                print("======{result.shape=}")
                output[seq_idx] = result.squeeze(0)  # [num_heads, head_size]

        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped
