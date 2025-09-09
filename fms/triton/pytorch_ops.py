from typing import Tuple

import torch
from torch.library import custom_op


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Args:
        topk_ids (torch.Tensor): A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
        block_size (int): The block size used in block matrix multiplication.
        num_experts (int): The total number of experts.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the sorted token ids, the expert to block mapping, and the total number of padded tokens.
    """

    # First count how many tokens go to each expert
    cnts = torch.zeros(
        topk_ids.shape[0], num_experts, dtype=topk_ids.dtype, device=topk_ids.device
    )
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)

    # Then pad the amount to the block size of work
    tokens_per_expert_block_padded = (
        torch.floor_divide(tokens_per_expert + block_size - 1, block_size) * block_size
    )

    # Count how many tokens in total are we computing in the GPU
    cumsum = tokens_per_expert_block_padded.cumsum(0)
    total_padded_tokens = cumsum[-1]

    # For allocation purposes, compute the worst case scenario for how
    # many tokens we could be doing work for if the RNG gods are bad
    max_total_padded_tokens = (
        (topk_ids.numel() + num_experts * (block_size - 1))
        if topk_ids.numel() > num_experts
        else (topk_ids.numel() + 1) * block_size
    )

    # Compute what MoE expert corresponds to each block of work
    # A block of work consists of tokens that all go to the same expert
    # There are total_padded_tokens // block_size blocks of work, but to
    # simplify kernel launches, we allocate based on worst case and use
    # max_total_padded_tokens instead.
    expert_block_mapping = torch.zeros(
        max((max_total_padded_tokens + block_size - 1) // block_size + 1, num_experts),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )

    num_blocks_per_expert = cumsum.div(block_size, rounding_mode="floor")
    ones = torch.ones_like(expert_block_mapping)
    expert_block_mapping.scatter_add_(0, num_blocks_per_expert, ones)
    expert_block_mapping = expert_block_mapping.cumsum(0)

    # Create the mapping between token idxs in the input tensor and
    # the tokens that will go in each work block

    # Count how many pad tokens need adding to the list of tokens in
    # topk_ids, then create a tensor that will fill in the right spots
    # when doing an argsort later to compute padded_token_ids_per_block
    cum_pad_tokens = (tokens_per_expert_block_padded - tokens_per_expert).cumsum(0)
    padded_tokens = torch.zeros(
        max_total_padded_tokens - topk_ids.numel(),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    ones = torch.ones_like(padded_tokens)
    padded_tokens.scatter_add_(0, cum_pad_tokens[:-1], ones)
    padded_tokens = padded_tokens.cumsum(0)
    padded_token_ids_per_block = torch.cat([topk_ids.view(-1), padded_tokens]).argsort()

    return padded_token_ids_per_block, expert_block_mapping, total_padded_tokens


@custom_op("moe::moe_mm", mutates_args={}, device_types="cuda")
def moe_mm(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
    padding_size: int,
) -> torch.Tensor:
    """
    A custom op for Mixture of Experts matrix multiplication.

    Args:
        input (torch.Tensor): The input tensor.
        moe_matrix (torch.Tensor): The MoE weight matrix.
        token_expert_mapping (torch.Tensor): A tensor mapping tokens to experts.
        padded_token_ids_per_block (torch.Tensor): A tensor of padded token IDs per block.
        expert_block_mapping (torch.Tensor): A tensor mapping experts to blocks.
        total_padded_tokens (torch.Tensor): The total number of padded tokens.
        topk (int): The number of top-k experts to use.
        padding_size (int): The padding size.

    Returns:
        torch.Tensor: The output tensor.
    """
    from fms.triton.moe_kernel import invoke_fused_moe_kernel

    M, A = token_expert_mapping.shape
    E, N, _ = moe_matrix.shape
    output = torch.zeros((M, A, N), device=input.device, dtype=input.dtype)

    invoke_fused_moe_kernel(
        input,
        moe_matrix,
        output,
        token_expert_mapping,
        padded_token_ids_per_block,
        expert_block_mapping,
        total_padded_tokens,
        topk,
        padding_size,
    )

    return output


@moe_mm.register_fake
def moe_mm_meta(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
    padding_size,
):
    """
    A fake implementation of the moe_mm custom op for meta tensors.

    Args:
        input (torch.Tensor): The input tensor.
        moe_matrix (torch.Tensor): The MoE weight matrix.
        token_expert_mapping (torch.Tensor): A tensor mapping tokens to experts.
        padded_token_ids_per_block (torch.Tensor): A tensor of padded token IDs per block.
        expert_block_mapping (torch.Tensor): A tensor mapping experts to blocks.
        total_padded_tokens (torch.Tensor): The total number of padded tokens.
        topk (int): The number of top-k experts to use.
        padding_size (int): The padding size.

    Returns:
        torch.Tensor: The output tensor.
    """
    M, A = token_expert_mapping.shape
    _, N, _ = moe_matrix.shape
    return torch.empty((M, A, N), device=input.device, dtype=input.dtype)


# TODO: Implement for real
def moe_mm_backward(ctx, grad_output):
    """
    The backward pass for the moe_mm custom op.

    Args:
        ctx: The context object.
        grad_output: The gradient of the output.

    Returns:
        Tuple: The gradients of the inputs.
    """
    (input_,) = ctx.saved_tensors
    # input, moe_matrix, token_expert_mapping, padded_token_ids_per_block, expert_block_mapping, total_padded_tokens
    return (
        input_,  # input
        None,  # moe_matrix
        None,  # token_expert_mapping
        None,  # padded_token_ids_per_block
        None,  # expert_block_mapping
        None,  # total_padded_tokens
        None,  # topk
        None,  # padding_size
    )


def moe_mm_setup_context(ctx, inputs, output):
    """
    Set up the context for the backward pass of the moe_mm custom op.

    Args:
        ctx: The context object.
        inputs: The inputs to the forward pass.
        output: The output of the forward pass.
    """
    (
        input_,
        moe_matrix,
        token_expert_mapping,
        padded_token_ids_per_block,
        expert_block_mapping,
        total_padded_tokens,
        topk,
        padding_size,
    ) = inputs
    ctx.save_for_backward(input_)
    pass


moe_mm.register_autograd(moe_mm_backward, setup_context=moe_mm_setup_context)


@moe_mm.register_kernel(["cpu", "mps"])
def moe_mm_cpu(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
    padding_size,
):
    """
    A CPU/MPS implementation of the moe_mm custom op.

    Args:
        input (torch.Tensor): The input tensor.
        moe_matrix (torch.Tensor): The MoE weight matrix.
        token_expert_mapping (torch.Tensor): A tensor mapping tokens to experts.
        padded_token_ids_per_block (torch.Tensor): A tensor of padded token IDs per block.
        expert_block_mapping (torch.Tensor): A tensor mapping experts to blocks.
        total_padded_tokens (torch.Tensor): The total number of padded tokens.
        topk (int): The number of top-k experts to use.
        padding_size (int): The padding size.

    Returns:
        torch.Tensor: The output tensor.
    """
    T, D = input.shape
    M, A = token_expert_mapping.shape

    a = input.view(T, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(T * topk, moe_matrix.shape[1], dtype=a.dtype, device=a.device)

    token_expert_mapping = token_expert_mapping.view(-1)
    for i in range(moe_matrix.shape[0]):
        mask = token_expert_mapping == i
        if mask.sum():
            out[mask] = a[mask] @ moe_matrix[i].transpose(0, 1)
    return out.view(M, A, moe_matrix.shape[1])
