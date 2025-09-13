# What I learnt

Problem 1
I first tried to derive the updates myself without reading the given flashattention pseudocode. What I was doing in my head was adjusting both the maximum and the l_i and then trying to calculate o updates by doing o * old_denominator/new_denominator. The actual algorithm does this normalization only towards the end. My intution was wait that is too late and maybe things will overflow but they actually don't.

Problem 3
Porting to triton wasn't so hard but i had to cast p_ij to bfloat16 when doing the accumlation for o. Clearly this kernel might break on super old gpus that don't support that. I should probably lookup how to detect if the GPU supports the architecture then select float16 or bloat16

Problem 4
First multiphase kernel. But the more important thing was that k and v blocks were not loaded for me. So i tried to not cheat and look at the previous problem and load myself. Added tons of comments to make sure shape and everything is correct. You can read them in the code.

Problem 5
Easy stuff no issues.

Problem 6
This was interesting because I realized that window_size and block_sizes don't have to be fully aligned which might require more complex masking. This could create problems with some parts just before the diagonal.

I tried to make the kernel more robust by handling it here

    # what happens if the window size is not a multiple of block size???
    # the below code handles it in absence of it we will not start at block boundaries and end up
    # skipping parts of the block just before the diagonal
    unaligned_window_start = tl.maximum(0, q_block_idx * BLOCK_M - WINDOW_SIZE)
    window_start = (unaligned_window_start // BLOCK_N) * BLOCK_N


Problem 7

I initally made the mistake of not taking care of the causal mask when I implemented the sinks. Important point because sinks are not supposed to behave like prefix lm masks.
