from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch

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
        tokenizer:PreTrainedTokenizerBase):
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).
    Args:
        prompt_strs(list[str]): List of prompt strings.
        output_strs(list[str]): List of output strings.
        tokenizer(PreTrainedTokenizer): Tokenizer to use for tokenization.
    Returns:
        output(dict[str, torch.Tensor]): Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings. Then the returned dictionary should have the following keys.
        - input_ids: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the tokenized prompt and output strings, with the final token sliced off.
        - labels: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): shifted input ids, i.e., the input ids without the first token.
        - response_mask: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): a mask on the response tokens in the labels.
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









import einops
from jaxtyping import Float, Int, Bool


# uv run pytest -k test_compute_entropy
def compute_entropy(
    logits: Float[torch.Tensor,"batch_size sequence_length vocab_size"]
) -> Float[torch.Tensor,"batch_size sequence_length"]:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits(torch.Tensor): Tensor of shape `(batch_size, sequence_length, vocab_size)` containing unnormalized logits.
    Returns:
        output(torch.Tensor): Shape `(batch_size, sequence_length)`. The entropy for each next-token prediction.
    """
    # 每个位置取对数，乘以自己的相反数，然后沿最后一维求和
    neglogp = - torch.log(logits)
    ent = einops.einsum(logits, neglogp, "b s v , b s v -> b s v")
    res = einops.reduce(ent, "b s v -> b s", "sum")
    return res