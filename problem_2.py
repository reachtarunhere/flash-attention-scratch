import torch
import triton
import triton.language as tl

@triton.jit
def weighted_row_sum_kernel(
    X_ptr,        # Pointer to the input tensor
    W_ptr,        # Pointer to the weight vector
    Y_ptr,        # Pointer to the output vector
    N_COLS,       # Number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for the kernel
):
    """
    Triton kernel to compute the weighted sum of each row in a matrix.
    Y[i] = sum_{j=0}^{N_COLS-1} X[i, j] * W[j]
    """
    # 1. Get the row index for the current program instance.
    #    Hint: Use tl.program_id(axis=0).
    row_idx = tl.program_id(axis=0)

    # 2. Create a pointer to the start of the current row in the input tensor X.
    #    Hint: The offset depends on the row index and the number of columns (N_COLS).
    row_start_ptr = X_ptr + row_idx * N_COLS
    
    # 3. Create a pointer for the output vector Y.
    output_ptr = Y_ptr + row_idx * N_COLS

    # 4. Initialize an accumulator for the sum of the products for a block.
    #    This should be a block-sized tensor of zeros.
    #    Hint: Use tl.zeros with shape (BLOCK_SIZE,) and dtype tl.float32.
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # 5. Iterate over the columns of the row in blocks of BLOCK_SIZE.
    #    Hint: Use a for loop with tl.cdiv(N_COLS, BLOCK_SIZE).
    for col_block_start in range(0, tl.cdiv(N_COLS, BLOCK_SIZE)):
        # - Calculate the offsets for the current block of columns.
        #   Hint: Start from the block's beginning and add tl.arange(0, BLOCK_SIZE).
        col_offsets = col_block_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        # - Create a mask to prevent out-of-bounds memory access for the last block.
        #   Hint: Compare col_offsets with N_COLS.
        mask = col_offsets < N_COLS
        
        # - Load a block of data from X and W safely using the mask.
        #   Hint: Use tl.load with the appropriate pointers, offsets, and mask.
        #   Use `other=0.0` to handle out-of-bounds elements.
        x_chunk = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        w_chunk = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        
        # - Compute the element-wise product and add it to the accumulator.
        accumulator += x_chunk * w_chunk
        # wait why not just dot these here huh?
        
    # 6. Reduce the block-sized accumulator to a single scalar value after the loop.
    #    Hint: Use tl.sum().
    final_sum = tl.sum(accumulator)

    # 7. Store the final accumulated sum to the output tensor Y.
    #    Hint: Use tl.store().
    tl.store(Y_ptr + row_idx, final_sum)
    # ...
    
# --- END OF STUDENT IMPLEMENTATION ---


def weighted_row_sum_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for the weighted row-sum operation using the Triton kernel.
    """
    assert x.dim() == 2, "Input tensor must be a 2D matrix"
    assert w.dim() == 1, "Weight tensor must be a 1D vector"
    assert x.shape[1] == w.shape[0], "Inner dimensions must match"
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA"
    
    N_ROWS, N_COLS = x.shape
    
    # The output is a 1D tensor with length equal to the number of rows.
    y = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
    
    # The grid is 1D, with one program instance per row.
    grid = (N_ROWS,)
    
    # Block size is a power of 2. 1024 is a good default.
    BLOCK_SIZE = 1024
    
    # Launch the kernel
    weighted_row_sum_kernel[grid](
        x, w, y,
        N_COLS=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y.to(x.dtype) # Cast back to original dtype

def torch_weighted_row_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using pure PyTorch.
    """
    return (x * w).sum(dim=1)
