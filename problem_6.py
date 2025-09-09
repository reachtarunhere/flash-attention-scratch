# import os

# os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl
import math
import ipdb


@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    # Stride information for tensors
    q_stride_b,
    q_stride_h,
    q_stride_s,
    k_stride_b,
    k_stride_h,
    k_stride_s,
    v_stride_b,
    v_stride_h,
    v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for Sliding Window Attention (SWA) with GQA.
    """
    # 1. Boilerplate setup
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 1: GQA Logic) ---
    # This problem combines GQA and SWA. First, implement the GQA logic.
    # 1. Calculate the number of query heads per group.
    # 2. Determine the correct kv_head_idx for the current q_head_idx.

    group_size = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    # --- END OF GQA IMPLEMENTATION ---

    # 2. Initialize accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load query block
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = (
        Q_ptr
        + batch_idx * q_stride_b
        + q_head_idx * q_stride_h
        + (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    )
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 2: SWA Logic) ---
    # Now, implement the "sliding window" by changing the loop bounds.
    # The kernel should only attend to the `WINDOW_SIZE` most recent key/value tokens.
    # 1. Calculate the starting position of the attention window (window_start).
    # 2. Modify the range of the Phase 1 loop to start from your window_start.

    # ipdb.set_trace()
    # first_query_row = q_block_idx * BLOCK_M
    # import ipdb

    # window_start = (
    #     first_query_row - WINDOW_SIZE
    # )  # Placeholder: Replace with your SWA calculation
    # window_start = tl.maximum(window_start, 0)

    # what happens if the window size is not a multiple of block size???
    # the below code handles it in absence of it we will not start at block boundaries and end up
    # skipping parts of the block just before the diagonal
    unaligned_window_start = tl.maximum(0, q_block_idx * BLOCK_M - WINDOW_SIZE)
    window_start = (unaligned_window_start // BLOCK_N) * BLOCK_N

    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        # 1. Load the K and V blocks for the current iteration.

        kv_start_row_idx = start_n

        # we need to load blocks of shape (HEAD_DIM, BLOCK_M) which will then be multiplied with Q blocks of
        # shape (BLOCK_N, HEAD_DIM)

        k_batch_head_base_ptr = (
            K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h
        )
        # Above basically gets us the ptr to K[batch_idx, head_idx, 0, 0]
        # now we do the same for v
        v_batch_head_base_ptr = (
            V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h
        )

        # the first part decides how many rows to skip
        row_offsets = kv_start_row_idx + tl.arange(0, BLOCK_N)
        k_offset_matrix = (
            row_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None]
        )
        # [HEAD_DIM, BLOCK_N]  =   [1, NUM_ROWS/BLOCK_N] +  [HEAD_DIM, 1]
        #                (contains start offsets of each k)   (column with 1..HEAD_DIM)

        k_block = tl.load(
            k_batch_head_base_ptr + k_offset_matrix,
            mask=row_offsets[None, :] < SEQ_LEN,
            other=0.0,
        )

        v_offset_matrix = (
            row_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :]
        )
        v_block = tl.load(
            v_batch_head_base_ptr + v_offset_matrix,
            mask=row_offsets[:, None] < SEQ_LEN,
            other=0.0,
        )

        # 2. Compute the attention scores (S_ij).

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        swa_mask = (q_offsets[:, None] - row_offsets[None, :]) < WINDOW_SIZE
        # if true we have to keep the values else mask
        s_ij = tl.where(swa_mask, s_ij, -float("inf"))

        # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).

        m_local = s_ij.max(axis=-1)
        m_new = tl.where(m_local > m_i, m_local, m_i)

        # Create a safe version of m_new to prevent -inf - (-inf)
        m_new = tl.where(m_new == -float("inf"), 0.0, m_new)

        # 2. Rescale the existing accumulator (`acc`) and denominator (`l_i`).
        scale_factor = tl.exp2(m_i - m_new)
        l_scaled = l_i * scale_factor
        acc_scaled = acc * scale_factor.expand_dims(1)

        # 3. Compute the attention probabilities for the current tile (`p_ij`).

        # unnormalized scores
        p_ij = tl.exp2(s_ij - m_new.expand_dims(1))

        # 4. Update the accumulator `acc` using `p_ij` and `v_block`.

        acc = acc_scaled + tl.dot(p_ij.to(tl.bfloat16), v_block)

        # 5. Update the denominator `l_i`.
        l_i = l_scaled + p_ij.sum(axis=1)
        # 6. Update the running maximum `m_i` for the next iteration.
        m_i = m_new

        # # STUDENT IMPLEMENTATION REQUIRED (Part 3: SWA Logic)
        # # Hint: You might need to apply the per-element sliding window mask to s_ij.
        # #    - A score is invalid if `(query_offset - key_offset) >= WINDOW_SIZE`.
        # pass

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        kv_start_row_idx = start_n

        # we need to load blocks of shape (HEAD_DIM, BLOCK_M) which will then be multiplied with Q blocks of
        # shape (BLOCK_N, HEAD_DIM)

        k_batch_head_base_ptr = (
            K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h
        )
        # Above basically gets us the ptr to K[batch_idx, head_idx, 0, 0]
        # now we do the same for v
        v_batch_head_base_ptr = (
            V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h
        )

        # the first part decides how many rows to skip
        row_offsets = kv_start_row_idx + tl.arange(0, BLOCK_N)
        k_offset_matrix = (
            row_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None]
        )
        # [HEAD_DIM, BLOCK_N]  =   [1, NUM_ROWS/BLOCK_N] +  [HEAD_DIM, 1]
        #                (contains start offsets of each k)   (column with 1..HEAD_DIM)

        k_block = tl.load(
            k_batch_head_base_ptr + k_offset_matrix,
            mask=row_offsets[None, :] < SEQ_LEN,
            other=0.0,
        )

        v_offset_matrix = (
            row_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :]
        )
        v_block = tl.load(
            v_batch_head_base_ptr + v_offset_matrix,
            mask=row_offsets[:, None] < SEQ_LEN,
            other=0.0,
        )

        # 2. Compute the attention scores (S_ij).

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        causal_mask = q_offsets[:, None] >= row_offsets[None, :]
        swa_mask = (q_offsets[:, None] - row_offsets[None, :]) < WINDOW_SIZE
        # Apply the mask to the attention scores
        s_ij = tl.where(causal_mask & swa_mask, s_ij, -float("inf"))

        # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).

        m_local = s_ij.max(axis=-1)
        m_new = tl.where(m_local > m_i, m_local, m_i)

        # Create a safe version of m_new to prevent -inf - (-inf)
        m_new = tl.where(m_new == -float("inf"), 0.0, m_new)

        # 2. Rescale the existing accumulator (`acc`) and denominator (`l_i`).
        scale_factor = tl.exp2(m_i - m_new)
        l_scaled = l_i * scale_factor
        acc_scaled = acc * scale_factor.expand_dims(1)

        # 3. Compute the attention probabilities for the current tile (`p_ij`).

        # unnormalized scores
        p_ij = tl.exp2(s_ij - m_new.expand_dims(1))

        # 4. Update the accumulator `acc` using `p_ij` and `v_block`.

        acc = acc_scaled + tl.dot(p_ij.to(tl.bfloat16), v_block)

        # 5. Update the denominator `l_i`.
        l_i = l_scaled + p_ij.sum(axis=1)
        # 6. Update the running maximum `m_i` for the next iteration.
        m_i = m_new
    # --- END OF SWA IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    o_ptrs = (
        O_ptr
        + batch_idx * q_stride_b
        + q_head_idx * q_stride_h
        + (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    )

    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]

    assert n_q_heads % n_kv_heads == 0, (
        "Number of query heads must be divisible by number of K/V heads"
    )
    assert is_causal, "This kernel is only supported for causal attention"

    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    # if window_size != 4096:
    #     raise ValueError("This kernel is compiled for a fixed window size of 4096")

    _flash_attention_forward_swa_kernel[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o


if __name__ == "__main__":
    q = torch.randn((2, 4, 512, 64)).to(torch.bfloat16)
    flash_attention_forward(
        q,
        q,
        q,
    )
    # print(q)
