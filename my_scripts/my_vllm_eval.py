from vllm import LLM, SamplingParams
from typing import List, Callable
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"
TEST_DATA_PATH = "cs336_alignment/test_data/test_data.jsonl"

def get_questions():
    return [
        "what is 1 + 1?",
        "Mary has 3 apples. She gives 1 apple to John. How many apples does Mary have left?",
    ]

def get_answers():
    return [
        "2",
        "2",
    ]

def get_prompt_template(path: str) -> str:
    with open(path, "r") as f:
        prompt_template = f.read()
    return prompt_template

def build_prompts(
    questions: List[str],
    prompt_template: str,
)-> List[str]:
    prompts = []
    for question in questions:
        prompt = prompt_template.format(question=question)
        prompts.append(prompt)
    return prompts

def get_sampling_params():
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

def generate_text(llm,prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers,
    eval_sampling_params: SamplingParams,
) -> None:
    """
    给定vllm模型和prompts，需要用给定的prompt模板进行拼接，然后送入llm进行推理，并使用reward_fn评估生成的文本，返回评估结果。
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for output, ground_truth in zip(outputs,answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        assert prompt != None, "Prompt should not be None"
        reward = reward_fn(generated_text,ground_truth)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Reward: {reward}")

if __name__ == "__main__":
    llm = LLM(model="models/Qwen2.5-Math-1.5B")
    prompts = build_prompts(get_questions(), get_prompt_template(PROMPT_TEMPLATE_PATH))
    answers = get_answers()
    
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=answers,
        eval_sampling_params=get_sampling_params(),
    )

