import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pass

@triton.jit
def _flash_attention_backward_swa_kernel(
    # In/Out Pointers
    Q_ptr, K_ptr, V_ptr, dO_ptr, M_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    d_stride_b, d_stride_h, d_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    # Parameters
    softmax_scale,
    BATCH_SIZE: int,
    N_Q_HEADS: int,
    N_KV_HEADS: int,
    SEQ_LEN: int,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Tile Sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pass

class FlashSWDAWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        assert is_causal, "Currently, only causal attention is supported"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape

        assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3], "Query and Value shapes must be compatible except for num_heads"
        assert k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3], "Key and Value shapes must be the same"
        assert head_dim <= 128, "Head dimension must be less than or equal to 128"
        assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"

        o = torch.empty_like(q)
        M = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)


        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        sink_size = ctx.sink_size

        assert do.is_contiguous()

        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        # Set up grid for the new q-centric backward kernel
        BLOCK_M, BLOCK_N = 64, 32
        grid_bwd = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)
        
        # TODO: Add your backward kernel here
        raise NotImplementedError("Backward kernel is left as an exercise for students!")

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None
    
def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)