import torch
import torch.nn.functional as F
from typing import Optional

def paged_attention_rocm_torch(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,          # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache: torch.Tensor,        # same
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,       # [batch_size, max_blocks]
    seq_lens: torch.Tensor,           # [batch_size]
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: Optional[torch.Tensor],
    partition_size: int,              # _PARTITION_SIZE_ROCM
):
    """
    Pure PyTorch implementation mimicking aiter.paged_attention_rocm.
    In-place writes to `output`.
    
    Assumptions:
      - key_cache/value_cache are in ROCm layout: [num_blocks, num_kv_heads, head_size, block_size]
      - All sequences fit in one partition (max_seq_len <= partition_size)
      - fp8_out_scale is ignored (output dtype = query.dtype)
    """
    # Validate assumptions
    assert max_seq_len <= partition_size, "Multi-partition not supported in this fallback"
    assert fp8_out_scale is None, "FP8 output not implemented"

    num_tokens, num_heads, head_size = query.shape
    batch_size = seq_lens.shape[0]
    device = query.device
    dtype = query.dtype

    gqa_ratio = num_heads // num_kv_heads
    assert num_heads % num_kv_heads == 0

    # We'll fill output in-place
    output.zero_()  # optional, but safe

    token_offset = 0
    for seq_id in range(batch_size):
        seq_len = seq_lens[seq_id].item()
        if seq_len == 0:
            continue

        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_ids = block_tables[seq_id, :num_blocks_needed]  # [n_blocks]

        # Gather K/V blocks: [n_blocks, num_kv_heads, head_size, block_size]
        k_blocks = key_cache[block_ids]
        v_blocks = value_cache[block_ids]

        # Dequantize if needed
        if kv_cache_dtype in ("fp8", "int8"):
            k_full = k_blocks.to(torch.float32) * k_scale.item()
            v_full = v_blocks.to(torch.float32) * v_scale.item()
        else:
            k_full = k_blocks.to(torch.float32)
            v_full = v_blocks.to(torch.float32)

        # Reshape to [num_kv_heads, head_size, total_len]
        k_full = k_full.permute(1, 2, 0, 3).reshape(num_kv_heads, head_size, -1)
        v_full = v_full.permute(1, 2, 0, 3).reshape(num_kv_heads, head_size, -1)

        # Trim to actual length
        k_full = k_full[:, :, :seq_len]  # [H_kv, D, S]
        v_full = v_full[:, :, :seq_len]  # [H_kv, D, S]

        # Transpose to [H_kv, S, D] for matmul
        k_full = k_full.transpose(-1, -2)  # [H_kv, S, D]
        v_full = v_full.transpose(-1, -2)  # [H_kv, S, D]

        # Expand for GQA
        if gqa_ratio > 1:
            k_full = k_full.unsqueeze(1).expand(-1, gqa_ratio, -1, -1).reshape(num_heads, seq_len, head_size)
            v_full = v_full.unsqueeze(1).expand(-1, gqa_ratio, -1, -1).reshape(num_heads, seq_len, head_size)
        else:
            k_full = k_full.expand(num_heads, -1, -1)
            v_full = v_full.expand(num_heads, -1, -1)

        # Query for this sequence: [S, H, D] -> [H, S, D]
        q_seq = query[token_offset:token_offset + seq_len].transpose(0, 1).to(torch.float32)  # [H, S, D]

        # Attention: [H, S, S]
        attn_weights = torch.matmul(q_seq, k_full.transpose(-1, -2)) * scale

        # ALiBi bias
        if alibi_slopes is not None:
            slopes = alibi_slopes[:num_heads].view(-1, 1, 1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            rel_pos = positions - positions.T
            alibi_bias = -slopes * rel_pos.abs()
            attn_weights += alibi_bias

        # Causal mask
        causal_mask = torch.triu(torch.full_like(attn_weights, float('-inf')), diagonal=1)
        attn_weights += causal_mask

        # Softmax
        attn_probs = F.softmax(attn_weights, dim=-1)

        # Output: [H, S, D]
        out = torch.matmul(attn_probs, v_full)

        # Write back to output buffer (in-place)
        out = out.transpose(0, 1).contiguous().to(dtype)  # [S, H, D]
        output[token_offset:token_offset + seq_len].copy_(out)

        # Update dummy buffers (for interface compatibility)
        if exp_sums is not None:
            # For single partition, exp_sum = sum(exp(logits - max_logit))
            max_logit_vals = attn_weights.max(dim=-1, keepdim=True).values  # [H, S, 1]
            exp_vals = torch.exp(attn_weights - max_logit_vals)
            exp_sum_vals = exp_vals.sum(dim=-1)  # [H, S]
            # But original kernel stores per-partition, we just fill with 1.0 for simplicity
            exp_sums[seq_id, :, :] = 1.0
            max_logits[seq_id, :, :] = 0.0

        token_offset += seq_len

    # tmp_output is unused in single-partition mode
    return
