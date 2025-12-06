from typing import Callable, Literal
import torch
from einops import einsum


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
)-> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
        reward_fn(Callable[[str, str], dict[str, float]]):
            Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses(list[str]):
            当前策略的一组输出回答。这些回答一共对应n_prompts_per_rollout_batch个不同的问题，而每个问题生成了group_size个不同的回答，故此列表总长度rollout_batch_size为二者之积。
        repeated_ground_truths(list[str]):
            每个问题的正确答案。因为每个问题生成了group_size个不同回答，为一一匹配，正确答案也重复了同样次数，故此列表总长度与rollout_responses同。
        group_size(int):
            对单个问题生成的不同回答数。
        advantage_eps(float):
            归一化分母中标准差所加的值，防止除零。
        normalize_by_std(bool):
            如果为True，则将结果除以标准差，否则只减去均值。

    Returns:
        output(tuple[torch.Tensor, torch.Tensor, dict[str, float]]):
            - advantages: shape (rollout_batch_size,). 各回答归一化的奖励（按要求决定是否除标准差）。
            - raw_rewards: shape (rollout_batch_size,). 各回答未归一化的奖励。
            - metadata: 此处自行记录其他统计量，如均值，标准差，最大/最小奖励等。
    """
    assert len(rollout_responses) % group_size == 0, "无法将所有输出正确分组"
    assert len(rollout_responses) == len(repeated_ground_truths), "生成回答和正确回答数量不同"

    all_count = len(rollout_responses)
    q_count = int(len(rollout_responses) / group_size)

    raw_rewards = []
    for i in range(all_count):
        reward = reward_fn(rollout_responses[i],repeated_ground_truths[i])
        raw_rewards.append(reward["reward"])
    raw_rewards = torch.tensor(raw_rewards) # 转为tensor
    
    pieces = []
    for i in range(q_count):
        rewards = raw_rewards[i*group_size:(i+1)*group_size].clone()
        rewards -= rewards.mean()
        if normalize_by_std == True:
            rewards /= (rewards.std() + advantage_eps)
        pieces.append(rewards)
    advantages = torch.cat(pieces)
    print("my advantages",advantages)
    print("my raw_rewards",raw_rewards)
    return (advantages, raw_rewards, {})
    #return (advantages,torch.tensor([]),{})
    #return(torch.tensor([]),torch.tensor([]),{})



# uv run pytest -k test_compute_naive_policy_gradient_loss
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    对于一批回答，给定各个回答的总体R或A值，计算每个回答的每个token对伪loss的贡献值。
    Args:
        raw_rewards_or_advantages(torch.Tensor): 形如(batch_size, 1)，每个回答的R或A
        policy_log_probs(torch.Tensor): 形如 (batch_size, sequence_length)，每个token的对数概率。

    Returns:
        output(torch.Tensor): 形如 (batch_size, sequence_length)，每个token对pgloss的贡献值 (to be aggregated across the batch and sequence dimensions in the training loop).
    """
    res = einsum(
        raw_rewards_or_advantages, 
        policy_log_probs,
        "b one, b s -> b s"
    )
    return -res



# uv run pytest -k test_compute_grpo_clip_loss
def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages(torch.Tensor):
            Shape (batch_size, 1), per-example advantages A.
        policy_log_probs(torch.Tensor):
            Shape (batch_size, sequence_length), per-token log probs from the policy being trained.
        old_log_probs(torch.Tensor):
            Shape (batch_size, sequence_length), per-token log probs from the old policy.
        cliprange(float): 
            Clip parameter ε (e.g. 0.2).

    Returns:
        output(tuple[torch.Tensor, dict[str, torch.Tensor]]):
            - loss: torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
            - metadata: dict containing whatever you want to log. We suggest logging whether each token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of the min was lower than the LHS.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = ratio.clip(1-cliprange,1+cliprange)
    res = - torch.min(ratio * advantages, clipped_ratio * advantages)
    return (res,{})


# uv run pytest -k test_compute_policy_gradient_loss
def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    从三个指定模式中选一，计算逐token的（伪）loss。

    Args:
        policy_log_probs (torch.Tensor):
            (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        loss_type (Literal):
            从 "no_baseline", "reinforce_with_baseline", "grpo_clip" 中三选一
        raw_rewards (torch.Tensor):
            Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages(torch.Tensor):
            Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
        old_log_probs(torch.Tensor):
            Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange(float):
            Required for "grpo_clip"; scalar ϵ used for clipping.

    Returns:
        output(tuple[torch.Tensor, dict[str, torch.Tensor]]):
            - loss: (batch_size, sequence_length), per-token loss.
            - metadata: dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "no_baseline模式需要raw_rewards"
        loss = compute_naive_policy_gradient_loss(raw_rewards,policy_log_probs)
        return (loss,{})
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "reinforce_with_baseline模式需要advantages"
        loss = compute_naive_policy_gradient_loss(advantages,policy_log_probs)
        return (loss,{})
    elif loss_type == "grpo_clip":
        assert advantages is not None \
            and old_log_probs is not None \
            and cliprange is not None, \
            "grpo_clip模式需要advantages，old_log_probs，cliprange"
        out = compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange)
        return out
    else:
        raise ValueError(f"没有{loss_type}这种模式！")
    

# uv run pytest -k test_masked_mean
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.

    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all
            masked elements.

    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    res = (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)
    return res 






def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs:
            (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        response_mask:
            (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps:
            Number of microbatches per optimizer step.
        loss_type:
            One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards:
            Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange Clip parameter ϵ for GRPO-Clip.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.
            metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    """