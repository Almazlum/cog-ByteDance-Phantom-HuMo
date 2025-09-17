# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """Robust attention with FlashAttention → PyTorch SDPA fallback.
    
    Tries FlashAttention v3/v2 for speed, falls back to torch.nn.functional.scaled_dot_product_attention
    if FlashAttention is unavailable or tensor shapes are incompatible.
    """
    out_dtype = q.dtype
    b, lq = q.size(0), q.size(1)
    if k_lens is not None:
        lk = k_lens.sum().item()
    else:
        lk = k.size(1)

    # FlashAttention kernels require fp16/bf16; cast if needed
    def _ensure_half_dtype(x):
        return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)

    if FLASH_ATTN_3_AVAILABLE:
        q = _ensure_half_dtype(q)
        k = _ensure_half_dtype(k)
        v = _ensure_half_dtype(v)
        # flashattention-v3
        x = flash_attn_interface.flash_attn_3(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        if not FLASH_ATTN_2_AVAILABLE:
            warnings.warn("Flash Attention not available, falling back to PyTorch scaled_dot_product_attention")
            attn_mask = None

            # Get dimensions
            b = q.size(0)
            num_heads = q.size(2)
            head_dim = q.size(3)

            # Already in [B, L, num_heads, head_dim] format
            # Move heads to dim 1 for torch.scaled_dot_product_attention
            q = q.transpose(1, 2)  # [B, num_heads, L, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)
                
            # Restore format to [B, L, num_heads, head_dim]
            out = out.transpose(1, 2)
            return out

        # flashattention-v2
        if q_lens is not None and k_lens is not None:
            q = _ensure_half_dtype(q)
            k = _ensure_half_dtype(k)
            v = _ensure_half_dtype(v)
            x = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic).unflatten(0, (b, lq))
        else:
            # Fallback when no sequence lengths or incompatible head counts
            # PyTorch SDPA handles MQA/GQA and various tensor layouts gracefully
            q_ = q.transpose(1, 2).to(torch.bfloat16)
            k_ = k.transpose(1, 2).to(torch.bfloat16)
            v_ = v.transpose(1, 2).to(torch.bfloat16)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=None, is_causal=causal, dropout_p=dropout_p
            )
            out = out.transpose(1, 2)
            return out.type(out_dtype)

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
