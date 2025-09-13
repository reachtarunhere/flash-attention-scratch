# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional
from causal_forward_kernel import _flash_attention_forward_gqa_kernel

class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        # TODO: Add your forward kernel here
        # we need to change the forward kernel to also support saving M which is L in flash attention paper

        _flash_attention_forward_gqa_kernel[grid](
        q,
        k,
        v,
        o,
        M,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        M.stride(0),
        M.stride(1),
        M.stride(2),        
        softmax_scale,
        seq_len,
        n_heads,
        n_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        )
        

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # [OPTIONAL BONUS] STUDENT IMPLEMENTATION REQUIRED
        # Implement the Triton backward kernel for GQA from scratch.
        # You should:
        #   1. Precompute delta = sum(dO * O)
        #   2. Recompute attention probabilities P = softmax(QK^T)
        #   3. Use delta + dO to accumulate gradients for dq, dk, dv
        #   4. Respect GQA mapping and causal mask
        
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)
