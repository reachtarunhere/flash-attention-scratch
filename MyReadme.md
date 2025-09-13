# Learnings from Implementing FlashAttention Variants

This document outlines the key insights and challenges encountered during the implementation of various attention mechanisms, from a basic PyTorch version to advanced Triton kernels.

## Problem 1: PyTorch Tiled Attention - Online Softmax

My initial attempt to implement the online softmax update for FlashAttention differed from the paper's algorithm. I tried to re-normalize the output `o` at each step by scaling it with the ratio of the old and new normalization denominators (`l_i`).

My intuition was that delaying the normalization until the end of the inner loop would risk numerical overflow. However, I learned that the FlashAttention algorithm is carefully designed to maintain numerical stability. The running maximum `m_i` is subtracted before the `exp` operation, which keeps the intermediate values in a safe range, preventing overflow. The final normalization is sufficient and more efficient.

## Problem 3: Porting to Triton

Porting the forward pass to Triton was a relatively smooth process. A key implementation detail was casting the `p_ij` matrix to `bfloat16` before accumulating into the output `o`.

This choice highlights a trade-off:
*   **Performance:** `bfloat16` offers significant performance gains on modern GPUs that support it.
*   **Portability:** The kernel will fail on older hardware that lacks `bfloat16` support.

A more robust implementation would dynamically check for hardware capabilities (e.g., via `torch.cuda.is_bf16_supported()`) and select either `bfloat16` or `float16` accordingly.

## Problem 4: Causal Masking in Triton

This was my first experience with a multi-phase Triton kernel. The template for this problem required me to implement the loading of `K` and `V` blocks myself, which proved to be a valuable exercise. It forced me to pay close attention to block pointer arithmetic and ensuring that tensor shapes and strides were correctly handled within the kernel, reinforcing my understanding of how Triton interacts with memory.

## Problem 5: Grouped-Query Attention (GQA)

Implementing GQA was a straightforward extension of the standard attention kernel. The core logic remained the same, with the only change being the mapping of multiple query heads to a single key/value head. This was achieved by adjusting the head indexing for the `K` and `V` tensors while keeping the `Q` head indexing unchanged.

## Problem 6: Sliding Window Attention (SWA) and Block Alignment

A subtle challenge in implementing SWA arose when the `window_size` was not an even multiple of the kernel's `BLOCK_SIZE`. A naive calculation of the window's start position for each block could misalign with the block boundaries, causing the kernel to skip over valid tokens just before the causal diagonal.

To address this, I implemented a more robust calculation for the window's starting position:

```python
# Handle cases where window_size is not a multiple of block size
unaligned_window_start = tl.maximum(0, q_block_idx * BLOCK_M - WINDOW_SIZE)
window_start = (unaligned_window_start // BLOCK_N) * BLOCK_N
```
This ensures that the window start is always aligned to a block boundary, preventing any tokens from being missed.

## Problem 7: SWA with Sinks and Causal Masking

When adding sink tokens to SWA, I initially made a conceptual error. I incorrectly assumed that sink tokens were exempt from causal masking entirely, similar to a prefix-LM mask.

The key learning here is that while sink tokens can be attended to by all subsequent tokens, **they themselves must still adhere to the causal mask**. A sink token at position `i` cannot attend to any token at position `j > i`. My final implementation correctly combines the sink mask with the causal mask to enforce this rule.


## Problem 8: Not Finished and Some Lessons

I spent maybe way too much time trying to derive the flash attention backward with all the jacobians by hand (I didn't want to cheat and look at the paper implementation). Overall it has been a good exercise. However I have decided to not yet implement coz I don't understand the concent of delta that is mentioned in the comments on the code. After looking at the flash attention paper I realized that it helps calculating dS from dP but without materalizing the full Jacobian. However, I am still not very clear how it avoids that so I will spend some more time with pen and paper before I go ahead and implement this.

dS( 𝑗) 𝑖 = P ( 𝑗) 𝑖 ◦ (dP( 𝑗) 𝑖 − 𝐷𝑖)

is the line that I didn't fully grasp.

Hopefully I can submit this later and still claim the bonus points as promised by Dr. Thang Luong :P
