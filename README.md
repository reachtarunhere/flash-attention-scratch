# 🚀 GStar Bootcamp: Building Flash Attention from Scratch

## Overview

In this assignment, you'll embark on a journey that takes you from the mathematical foundations of attention mechanisms to implementing state-of-the-art Flash Attention kernels that power today's most advanced language models.

By the end of this assignment, you'll have built Flash Attention and its variants **completely from scratch** using both PyTorch and Triton, giving you deep understanding of how to optimize attention mechanisms at the GPU level. You're about to learn how this magic works - not just the theory, but by implementing every line of code yourself.

## 📄 Bootcamp Companion PDF

**For detailed background, tutorials, and task descriptions, please see the accompanying PDF file.** This document is your primary guide for the theory and implementation details needed for each stage.

## What You'll Achieve

By the end of this bootcamp, you'll have built Flash Attention from scratch, gaining a deep understanding of how and why it works—both its memory efficiency and mathematical exactness. You will also develop strong technical skills: writing efficient GPU kernels in Triton.

Furthermore, you will see the real-world impact of your work by implementing the same algorithms that power production AI systems such as [GPT-OSS](https://openai.com/index/introducing-gpt-oss/), [Llama](https://huggingface.co/meta-llama), and [Gemma](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d). You'll understand the optimization techniques that let modern LLMs handle long contexts or streaming conversations, and **you will be equipped with the foundation to contribute to next-generation AI infrastructure.**



### **SOTA Mastery** (Advanced Stages)
- **Group Query Attention (GQA)**: A critical memory optimization technique that has become standard in modern large language models, including the Llama series, Gemma, GPT-OSS.
- **Sliding Window**: A technique for efficiently processing long sequences by limiting the attention scope of each token to a fixed-size window.
- **Attention Sinks**: A recent breakthrough that stabilizes model performance when dealing with streaming inputs. Recently adopted by [GPT-OSS].
- **Backward Pass Implementation** (Optional): Deep understanding of gradient computation, recomputation, and autograd systems.


## Your Four-Stage

### Stage 1: The Foundation (PyTorch)
**File**: `problem_1.py` - Implement Flash Attention in pure PyTorch. Learn tiling, online softmax, and why it's both efficient and exact.

### Stage 2: Your First GPU Kernel (Triton)
**File**: `problem_2.py` - Write your first GPU kernel with weighted row sums. Grasp parallel thinking, memory access patterns, and block-based efficiency.

### Stage 3: Flash Attention Unleashed (Triton)
**File**: `problem_3.py` — Turn your PyTorch version into a high-performance GPU kernel. Master online softmax at scale and reach production-level speed.

### Stage 4: The Causal Challenge (Triton)
**File**: `problem_4.py` — Tackle causal masking in Large Language Models with two-phase kernels and tiled computation 

## Advanced Challenges (SOTA Variants)

These advanced stages implement important optimizations used in modern AI systems:

### Stage 5: Grouped Query Attention (GQA)
**File**: `problem_5.py` - Implement the K/V sharing trick behind Llama, Mistral, and others. Reduce memory significantly when scaling to billions of parameters.


### Stage 6 + 7: Sliding Window Attention (SWA) and Attention Sinks
**File**: `problem_6.py & problem_7.py` - Build local attention windows (with attention sinks) for infinite context

### Stage 8 (Optional) : Backward Pass for Grouped Query Attention 
**File**: `problem_8.py` - — Implement the backward pass for GQA to enable fine-tuning. Explore autograd internals, apply recomputation strategies, and optimize for memory-efficient backpropagation. 

### Research Challenge (Stage 9) (Optional): The Missing Piece for [GPT-OSS](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) Finetuning
**File** `problem_9.py` - This is a research-level (challenging) problem that addresses a real gap in today's ecosystem: enabling finetuning with Flash Attention when combined with Sliding Window and Attention Sinks.

Since GPT-OSS release, people have not been able to finetune it with Flash Attention. OpenAI only provides the forward pass for inference. For training, the [OpenAI Cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) and other resources fall back to naive attention, since the backward pass with Flash Attention is missing. Even official `flash_attn` library and related implementations currently lack support for finetuning with Flash Attention + Sliding Attention + Attention Sinks.

**Your challenge**: If you can solve problem_8, you will have the foundation to attempt problem_9: implementing the missing backward pass for GPT-OSS finetuning.


## Prerequisites & Setup

You'll need:
- Python 3.8+
- PyTorch with CUDA support
- Triton library
- A CUDA-capable GPU

### Quick Setup
```bash
# Install dependencies
pip install torch triton

# Verify your GPU is ready
python -c "import torch; print(f'CUDA ready: {torch.cuda.is_available()}')"
```

## File Structure

```
GStar/
├── README.md                 # This file
├── problem_1.py             # Stage 1: PyTorch FlashAttention (YOUR CODE HERE)
├── problem_2.py             # Stage 2: Triton Weighted Row-Sum (YOUR CODE HERE)
├── problem_3.py             # Stage 3: Triton FlashAttention Non-causal (YOUR CODE HERE)
├── problem_4.py             # Stage 4: Triton FlashAttention Causal (YOUR CODE HERE)
├── problem_5.py             # Stage 5: Grouped Query Attention (YOUR CODE HERE)
├── problem_6.py             # Stage 6: Sliding Window Attention (YOUR CODE HERE)
├── problem_7.py             # Stage 7: Attention Sinks (YOUR CODE HERE)
├── problem_8.py             # Optional: GQA Backward Pass (BONUS CHALLENGE)
├── problem_9.py             # Optional: GQA + SWDA + Attention Sinks Backward Pass (MORE CHALLENGE)
├── autograder.py            # Automated testing and benchmarking
├── autograder_optional.py   # Autograder for Stages 8 & 9
└── .gitignore              # Git ignore file
```

## 💻 How to Work with Problem Files

Each `problem_X.py` contains:
- **Background and Detailed problem description** with mathematical background
- **Function signatures** you need to implement
- **Template code** with helpful comments and hints
- **TODO markers** showing exactly where to add your code


**Your Task**: Replace the `TODO` sections and `pass` statements with working implementations!

## 🎯 Getting Started

### Start with Stage 1
Begin your journey with `problem_1.py` - it builds the foundation for everything else:

```bash
# Open the first problem file
code problem_1.py

# Read through the problem description and understand what you need to implement
# Look for TODO markers and function signatures
# Start implementing step by step

# Test your progress frequently
python autograder.py --p1
```
## Testing Your Implementations

Your implementation journey is guided by a comprehensive autograder that validates correctness and measures performance against reference implementations.

### Stage-by-Stage Testing (Problems 1–7)

```bash
python autograder.py --p1   # PyTorch baseline
python autograder.py --p2   # First Triton kernel
python autograder.py --p3   # FlashAttention (non-causal)
python autograder.py --p4   # FlashAttention (causal)
python autograder.py --p5   # Flash Group Query Attention
python autograder.py --p6   # Flash Group Query Attention + Sliding Window Attention
python autograder.py --p7   # Flash Group Query Attention + Sliding Window Attention & Attention Sinks
```

#### Optional Challenges (Problems 8–9)
```bash
python autograder_optional.py --p8   # Backward pass for GQA
python autograder_optional.py --p9   # Backward pass for GQA + SWA + Attention Sinks
```
###  Test Output

The autograder compares your implementation against mathematically correct reference implementations and provides detailed feedback:

- **Correctness**: Your implementation must match the mathematical ground truth
- **Performance**: Speed benchmarks show how your optimizations stack up
- **Memory**: Efficiency metrics reveal your memory optimization gains  

**What success looks like:**
```
--- Running Autograder for Problem 3: Non-Causal Flash Attention ---
✅ P3 Correctness Test Passed! (B=1, H=8, L=512, D=16)
✅ P3 Correctness Test Passed! (B=1, H=8, L=1024, D=16)
✅ P3 Correctness Test Passed! (B=1, H=16, L=2048, D=16)
✅ P3 Correctness Test Passed! (B=1, H=16, L=4096, D=16)

All P3 correctness tests passed!

--- Running Performance Benchmark ---
Benchmark Config: B=1, H=16, L=4096, D=16, Causal=False

--- Benchmark Results ---
Implementation            | Avg Time (ms)        | Peak Memory (GB)    
----------------------------------------------------------------------
PyTorch (Naive)           | 10.6512              | 3.0162              
Triton (Flash)            | 0.2424               | 0.0157              
----------------------------------------------------------------------
Triton is 43.94x faster than PyTorch (Naive).
Triton uses 191.54x less memory.
```

*** Example output for Problem 8+9: ***
```
--- Running Autograder for Problem 9: GQA + SWDA + Attention Sinks Backward Pass ---
Running test case: batch=1, heads_q=16, heads_kv=1, seq_len=4096, dim=16, window_size=256, sink_size=4
✅ Forward Pass Results match
✅ Backward Pass Results match on dQ
✅ Backward Pass Results match on dK
✅ Backward Pass Results match on dV
```

## Essential Resources

These resources will support your hands-on learning journey:

- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691) - Study the algorithmic insights you'll implement
- [Triton Documentation](https://triton-lang.org/) - Your GPU kernel programming reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The mathematical foundation