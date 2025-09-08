import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attention_forward_gqa_kernel(
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
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for the forward pass of causal FlashAttention with GQA.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)

    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 1) ---
    # Your goal is to map the current query head (q_head_idx) to its corresponding shared key/value head (kv_head_idx).
    # 1. Calculate how many query heads are in each group.
    # 2. Use integer division to find the correct kv_head_idx.

    group_size = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size  # Placeholder: Replace with your calculation
    # --- END OF STUDENT IMPLEMENTATION ---

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = (
        Q_ptr
        + batch_idx * q_stride_b
        + q_head_idx * q_stride_h
        + (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    )
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    qk_scale = softmax_scale * 1.44269504

    # --- Phase 1: Off-Diagonal Blocks ---
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
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

        # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).

        m_local = s_ij.max(axis=-1)
        m_new = tl.where(m_local > m_i, m_local, m_i)

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
        # Apply the mask to the attention scores
        s_ij = tl.where(causal_mask, s_ij, -float("inf"))

        # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).

        m_local = s_ij.max(axis=-1)
        m_new = tl.where(m_local > m_i, m_local, m_i)

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

        # --- END OF STUDENT IMPLEMENTATION ---
    # 4. Normalize and write the final output block.
    l_i_safe = l_i[:, None] + 1e-6
    acc = acc / l_i_safe

    o_ptrs = (
        O_ptr
        + batch_idx * q_stride_b
        + q_head_idx * q_stride_h
        + (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    )

    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True):
    """
    Python wrapper for the GQA-enabled causal FlashAttention kernel.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]

    assert n_q_heads % n_kv_heads == 0, (
        "Number of query heads must be divisible by number of K/V heads"
    )

    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_gqa_kernel[grid](
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
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o
