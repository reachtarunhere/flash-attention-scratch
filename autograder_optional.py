import sys
import argparse

import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def create_mask_bool(
    seq_len: int,
    window_size: int,
    sink_size: int,
    device=None
    ) -> torch.Tensor:
    
    idx = torch.arange(seq_len, device=device)
    row = idx.unsqueeze(1)
    col = idx.unsqueeze(0)

    sliding = (col <= row) & (col >= row - (window_size - 1))
    sink = (col < sink_size) & (col <= row)

    return sliding | sink

def naive_attention(q, k, v, seq_len, window_size, sink_size):
    return F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=create_mask_bool(seq_len, window_size, sink_size, device=q.device),
        enable_gqa=True,
    )
    
    
def check_backward_correctness(triton_func, problem_num):
    test_cases = [
        (1, 16, 16, 4096, 16, 256, 4),
        (1, 16, 8, 4096, 16, 256, 4),
        (1, 16, 1, 4096, 16, 256, 4),
    ]
    for case in test_cases:
        batch, heads_q, heads_kv, seq_len, dim, window_size, sink_size = case
        
        if problem_num == 8:
            print(f"Running test case: batch={batch}, heads_q={heads_q}, heads_kv={heads_kv}, seq_len={seq_len}, dim={dim}")
        elif problem_num == 9:
            print(f"Running test case: batch={batch}, heads_q={heads_q}, heads_kv={heads_kv}, seq_len={seq_len}, dim={dim}, window_size={window_size}, sink_size={sink_size}")
        else:
            raise ValueError(f"Problem {problem_num} not supported")
        
        q = torch.randn(batch, heads_q, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
        k = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
        v = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
        
        q_ref, k_ref, v_ref = q.clone().detach().requires_grad_(), k.clone().detach().requires_grad_(), v.clone().detach().requires_grad_()
        
        if problem_num == 8:
            o_ref = naive_attention(q_ref, k_ref, v_ref, seq_len=seq_len, window_size=seq_len, sink_size=0)
            o_triton = triton_func(q, k, v, is_causal=True)
        elif problem_num == 9:
            o_ref = naive_attention(q_ref, k_ref, v_ref, seq_len, window_size, sink_size)
            o_triton = triton_func(q, k, v, window_size=window_size, sink_size=sink_size, is_causal=True)
        else:
            raise ValueError(f"Problem {problem_num} not supported")
            
        
        is_forward_correct = torch.allclose(o_ref, o_triton, atol=1e-2, rtol=1e-2)
        if is_forward_correct:
            print("✅ Forward Pass Results match")
        else:
            print("❌ Forward Pass Results do not match")
        
        dout = torch.rand_like(o_ref)
        o_ref.backward(dout)
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad
        
        o_triton.backward(dout)
        dq_flash, dk_flash, dv_flash = q.grad, k.grad, v.grad
        
        is_dq_correct = torch.allclose(dq_ref, dq_flash, atol=5e-2, rtol=5e-2)
        is_dk_correct = torch.allclose(dk_ref, dk_flash, atol=5e-2, rtol=5e-2)
        is_dv_correct = torch.allclose(dv_ref, dv_flash, atol=5e-2, rtol=5e-2)
        if is_dq_correct:
            print("✅ Backward Pass Results match on dQ")
        else:
            print("❌ Backward Pass Results do not match on dQ")
        if is_dk_correct:
            print("✅ Backward Pass Results match on dK")
        else:
            print("❌ Backward Pass Results do not match on dK")
        if is_dv_correct:
            print("✅ Backward Pass Results match on dV")
        else:
            print("❌ Backward Pass Results do not match on dV")


def check_problem_8():
    """Checks Problem 8: GQA."""
    problem_num = 8
    print(f"\n--- Running Autograder for Problem {problem_num}: GQA Backward Pass ---")
    try:
        from problem_8 import flash_attention_gqa
    except ImportError:
        print(f"Could not import FlashAttention2Function from solution_{problem_num}.py.")
        return
    
    torch.manual_seed(48)
    check_backward_correctness(flash_attention_gqa, problem_num)
    

def check_problem_9():
    """Checks Problem 9: GQA + SWDA + Attention Sinks Backward Pass."""
    problem_num = 9
    print(f"\n--- Running Autograder for Problem {problem_num}: GQA + SWDA + Attention Sinks Backward Pass ---")
    try:
        from problem_9 import flash_swda_with_sink
    except ImportError:
        print(f"Could not import FlashAttention2Function from solution_{problem_num}.py.")
        return
    
    torch.manual_seed(48)
    check_backward_correctness(flash_swda_with_sink, problem_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autograder for Triton Flash Attention assignments.")
    parser.add_argument('--p8', action='store_true', help='Run autograder for Problem 8 (GQA Backward Pass).')
    parser.add_argument('--p9', action='store_true', help='Run autograder for Problem 9 (GQA + SWDA + Attention Sinks Backward Pass).')
    
    if not torch.cuda.is_available():
        print("💥 CUDA not available. Skipping all GPU tests.")
        sys.exit(1)

    args = parser.parse_args()
    if not any(vars(args).values()):
        args.p8 = args.p9 = True
    
    if args.p8:
        check_problem_8()
    if args.p9:
        check_problem_9()