from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
import torch

import einops
from jaxtyping import Float, Int, Bool

#uv run pytest -k test_tokenize_prompt_and_output
def get_mask_tensor(io_len:list[tuple[int,int]])->torch.Tensor:
    """
    根据已知的io长度制造mask，只显露出模型输出的部分，mask掉输入和尾部padding
    """
    max_len = max([i + o for (i,o) in io_len])
    res = []
    for ilen, olen in io_len:
        #print("i,o:",ilen,olen)
        imask = [0] * ilen
        omask = [1] * olen
        padmask = [0] * (max_len - ilen - olen)
        res.append(imask + omask + padmask)
    rest = torch.tensor(res)
    return rest



def tokenize_prompt_and_output(
        prompt_strs:list[str],
        output_strs:list[str], 
        tokenizer:PreTrainedTokenizerBase,
)->dict[str,torch.Tensor]:
    """
    将给定的问题列表和回答列表制成一份可用于sft的训练数据，包括相互错开一个token的input-label对应表；以及一个mask，用1标出label中回答部分的位置，而问题和尾部填充部分标0。使用给定的分词器。

    在本实现中，尾部填充使用`tokenizer.pad_token_id`。
    Args:
        prompt_strs(list[str]): 问题列表。
        output_strs(list[str]): 回答列表。
        tokenizer(PreTrainedTokenizer): 给定的分词器。
    Returns:
        output(dict[str, torch.Tensor]): 将token数最多的“问题+回答”组合含有的token数记为L，则返回字典应包含如下内容：
        - input_ids: torch.Tensor，形如(batch_size, L - 1): 所有“问题+回答”对的分词结果，去掉最后一个token。
        - labels: torch.Tensor，形如(batch_size, L - 1): 所有“问题+回答”对的分词结果左错一位，即去掉首个token。
        - response_mask: torch.Tensor，形如(batch_size, L - 1): 在labels上标出“回答”部分位置的mask。
    """
    # print("输入:",prompt_strs)
    # print("输出:",output_strs)

    assert len(prompt_strs) == len(output_strs) , "输入与输出数量不等！"
    
    prompt_ids = [tokenizer.encode(s) for s in prompt_strs]
    #print(prompt_ids)

    response_ids = [tokenizer.encode(s) for s in output_strs]
    #print(response_ids)
    
    batch_ids = [p + r for p,r in zip(prompt_ids, response_ids)]
    #print(batch_ids)

    io_len = [(len(lp),len(lr)) for lp, lr in zip(prompt_ids, response_ids)]
    #print(io_len)

    max_len = max([len(io) for io in batch_ids])
    #print("最长的序列长度：",max_len)

    padding_id = tokenizer.pad_token_id
    #print("padding id:",padding_id)
    #print(tokenizer.decode(padding_id))

    for io in batch_ids:
        l = len(io)
        pad = [padding_id for _ in range(max_len - l)]
        io += pad

    #print(batch_ids)

    



    padded_batch_tensor = torch.tensor(batch_ids)
    #print(padded_batch_tensor)

    mask_tensor = get_mask_tensor(io_len)
    #print(mask_tensor)

    res =  {
        "input_ids":padded_batch_tensor[:,:-1],
        "labels":padded_batch_tensor[:,1:],
        "response_mask":mask_tensor[:,1:]
    }

    return res



# uv run pytest -k test_compute_entropy
def compute_entropy(
    logits: Float[torch.Tensor,"batch_size sequence_length vocab_size"]
) -> Float[torch.Tensor,"batch_size sequence_length"]:
    """
    给定一组logits，计算其对应概率分布的熵。默认前置维度包括b和s。实现中考虑了数值稳定性，避免softmax分母上溢。
    Args:
        logits(torch.Tensor): 形如`(batch_size, sequence_length, vocab_size)`，内含logits。
    Returns:
        output(torch.Tensor): 形如`(batch_size, sequence_length)`，对每组logits对应的概率分布求出其熵。
    """

    max_logits = einops.reduce(logits,"b s v -> b s 1","max")   # b s 1
    logits -= max_logits                                        # b s v
    # 统一减去最大值，此后只使用这种形式计算
    
    exp = torch.exp(logits)                                     # b s v
    sumexp = einops.reduce(exp,"b s v -> b s 1","sum")          # b s 1
    prob = exp / sumexp                                         # b s v

    logsumexp = torch.log(sumexp)                               # b s 1
    logprob = logits - logsumexp                                # b s v

    entropy_contrib = einops.einsum(-prob,logprob,"b s v, b s v -> b s v")
    entropy = einops.reduce(entropy_contrib,"b s v -> b s","sum")

    return entropy



def compute_prob_given_id(
    logits: Float[torch.Tensor,"batch_size sequence_length vocab_size"],
    id: Int[torch.Tensor,"batch_size sequence_length"],
) -> Float[torch.Tensor,"batch_size sequence_length"]:
    """
    给定一组logits，计算序号为id的token经softmax所得概率。默认前置维度包括b和s。实现中考虑了数值稳定性，避免softmax分母上溢。
    Args:
        logits(torch.Tensor): 形如`(batch_size, sequence_length, vocab_size)`，内含logits。
        id(torch.Tensor): 形如`(batch_size, sequence_length)`，为每组logits指明要计算概率的位置。
    Returns:
        output(torch.Tensor): 形如`(batch_size, sequence_length)`，记录每组logits的第id个token对应的softmax所得概率。
    """

    max_logits = einops.reduce(logits,"b s v -> b s 1","max")   # b s 1
    logits -= max_logits                                        # b s v
    # 统一减去最大值，此后只使用这种形式计算
    
    exp = torch.exp(logits)                                     # b s v
    sumexp = einops.reduce(exp,"b s v -> b s 1","sum")          # b s 1
    logsumexp = torch.log(sumexp)                               # b s 1
    logprob = logits - logsumexp                                # b s v

    res = torch.gather(logprob, dim = 2, index = id.unsqueeze(-1)).squeeze(-1)

    return res


# 测试：uv run pytest -k test_get_response_log_probs
def get_response_log_probs(
    model,#: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    给定一组input-label，考察一个（因果）模型对每个input位置是否能正确预测出label。即：计算当前模型参数下，对于每个input位置，模型根据从头到该位置的input预测下一个token恰是该位置对应label的概率。顺便提供同时返回模型在每个位置预测的概率分布的熵的功能。
    Args:
        model(PreTrainedModel): 待评价的模型。(placed on the correct device and in inference mode if gradients should not be computed).
        input_ids(torch.Tensor): 形如 `(batch_size, sequence_length)`，来自多组“问题+回答”对（去尾）。
        labels(torch.Tensor): 形如 `(batch_size, sequence_length)`，来自多组“问题+回答”对（左移一位，即去头）。
        return_token_entropy(bool): 若为 `True`，顺便同时返回每个位置预测分布的熵。应调用`compute_entropy`。
    Returns:
        output(dict[str, torch.Tensor]):
        - "log_probs": 形如 `(batch_size, sequence_length)`，即条件概率 $log p_θ(x_t | x_{<t})$.
        - "token_entropy": 可选，形如 `(batch_size, sequence_length)`，每个位置预测分布的熵 (只有`return_token_entropy=True`时返回此项).
    """
    # model(input_ids).logits 可以用于获得logits
    logits = model(input_ids).logits
    # print(logits)
    # print(logits.size())
    # print(labels)
    # print(labels.size())
    res = {}
    res["log_probs"] = compute_prob_given_id(logits,labels)
    print(res["log_probs"].size())
    if return_token_entropy == True:
        res["token_entropy"] = compute_entropy(logits)
    return res


# uv run pytest -k test_masked_normalize
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where `mask == 1`.
    Args:
        tensor(torch.Tensor): The tensor to sum and normalize.
        mask(torch.Tensor): Same shape as `tensor`; positions with `1` are included in the sum.
        normalize_constant(float): the constant to divide by for normalization.
        dim(int | None): the dimension to sum along before normalization. If `None`, sum over all dimensions.
    Returns:
        output(torch.Tensor): the normalized sum, where masked elements (`mask == 0`) don’t contribute to the sum.
    """
    s = (tensor * mask).sum(dim=dim) / normalize_constant
    return s 
    

# uv run pytest -k test_sft_microbatch_train_step
def sft_microbatch_train_step(
    policy_log_probs: Float[torch.Tensor,"batch_size sequence_length"],
    response_mask: Float[torch.Tensor,"batch_size sequence_length"],
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs(Float[torch.Tensor,"batch_size sequence_length"]): 当前policy每个位置的logprob（其实就是交叉熵的相反数）
        response_mask(Float[torch.Tensor,"batch_size sequence_length"]): 用1标出回复部分，问题和填充部分为0，最终只把mask部分加起来
        gradient_accumulation_steps(int): 梯度累积的步数
        normalize_constant(float): 归一化常数，默认为1
    Returns:
        output(tuple[torch.Tensor, dict[str, torch.Tensor]]):
        - loss: 单元素张量，loss（根据梯度累积调整过，即除以步数）
        - metadata: Dict with metadata from the underlying loss call, and any other statistics you might want to log.
    """
    b, l = policy_log_probs.size()
    loss = - masked_normalize(
        policy_log_probs, 
        response_mask, normalize_constant=normalize_constant,
        dim = None,
    ) / (gradient_accumulation_steps * b)
    # 为什么要对b平均？但总之这样是对的
    loss.backward()
    return loss.detach(), {}