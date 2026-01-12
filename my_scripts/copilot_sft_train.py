import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to sys.path to allow imports from cs336_alignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_alignment.my_sft_utils import (
    tokenize_prompt_and_output,
    sft_microbatch_train_step,
    masked_normalize,
    get_response_log_probs
)
from cs336_alignment.my_sft_toolfunc import read_jsonl, get_qa_list
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Try importing wandb, but don't crash if it's missing (though it's expected)
try:
    import wandb
except ImportError:
    wandb = None

# =============================================================================
# 0. Helper Functions & Classes
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_prompt_template(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_gsm8k_answer(answer_str):
    """
    Parses GSM8K answer string.
    Format: "Explanation... #### FinalAnswer"
    Returns: (explanation, final_answer)
    """
    parts = answer_str.split("####")
    if len(parts) >= 2:
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
        return reasoning, final_answer
    else:
        return answer_str.strip(), ""

def format_data_for_r1_zero(data_list, prompt_template):
    """
    Applies the R1-Zero prompt template to questions and formats answers.
    Target Answer Format: "{Reasoning} </think> <answer> {FinalAnswer} </answer>"
    """
    formatted_data = []
    for item in data_list:
        question = item["question"]
        raw_answer = item["answer"]
        
        # Format Question
        full_prompt = prompt_template.format(question=question)
        
        # Format Answer
        reasoning, final_answer = parse_gsm8k_answer(raw_answer)
        # Note: The prompt ends with "Assistant: <think>", so the model should continue with the reasoning content.
        # We do NOT include the opening <think> in the target because it's already in the prompt (or implied).
        # Wait, let's check exact prompt content provided in attachments.
        # Attachment says: "Assistant: <think>\n" (or similar). 
        # So the model generation starts INSIDE the think block.
        # So we expect: "Reasoning... </think> <answer> FinalAnswer </answer>"
        
        formatted_answer = f"{reasoning} </think> <answer> {final_answer} </answer>"
        
        formatted_data.append({
            "question": full_prompt,
            "answer": formatted_answer,
            "ground_truth": final_answer # Keep raw ground truth for eval if needed
        })
    return formatted_data

class GSM8KDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloaders(args, tokenizer):
    print(f"Loading data from {args.train_data}...")
    
    # Read Prompt Template
    prompt_template = load_prompt_template(args.prompt_template_path)
    
    # Load and Process Data
    train_raw = list(read_jsonl(args.train_data))
    if args.max_train_samples:
        train_raw = train_raw[:args.max_train_samples]
    
    train_formatted = format_data_for_r1_zero(train_raw, prompt_template)
    
    # We do NOT create a validation loader for SFT loss here, 
    # because the user specifically asked for "vLLM evaluation" every 0.5 epoch.
    # But keeping a small validation set for loss monitoring is good practice.
    # However, to strictly follow instructions, we focus on the vLLM eval.
    # We will just return the train loader.
    
    print(f"Training samples: {len(train_formatted)}")
    
    def collate_fn(batch):
        prompts = [item["question"] for item in batch]
        responses = [item["answer"] for item in batch]
        
        encoded_batch = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer
        )
        return encoded_batch

    train_loader = DataLoader(
        GSM8KDataset(train_formatted),
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    return train_loader

# =============================================================================
# 1. Evaluation Logic (Evaluation Mode)
# =============================================================================

def run_evaluation_mode(args):
    """
    Independent evaluation routine using vLLM.
    This is called when the script is run with --eval_only
    """
    from vllm import LLM, SamplingParams
    # Import LoRARequest if LoRA is used
    from vllm.lora.request import LoRARequest
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    print(f"Starting vLLM Evaluation on {args.eval_device}...")
    
    # Load Test Data
    test_raw = list(read_jsonl(args.test_data))
    if args.max_eval_samples:
        test_raw = test_raw[:args.max_eval_samples]
    
    prompt_template = load_prompt_template(args.prompt_template_path)
    
    questions = [prompt_template.format(question=item["question"]) for item in test_raw]
    ground_truths = [parse_gsm8k_answer(item["answer"])[1] for item in test_raw]

    # Initialize vLLM
    # If using LoRA, we init the BASE model, and pass the adapter in generation
    # If NOT using LoRA, args.model_path is the full model checkpoint
    
    if args.use_lora and args.base_model_path:
        print(f"Loading Base Model for LoRA: {args.base_model_path}")
        llm = LLM(
            model=args.base_model_path,
            enable_lora=True,
            tensor_parallel_size=1,
            device=args.eval_device, # parses 'cuda:0' -> 'cuda' usually, explicit device setting via environment might be safer
            gpu_memory_utilization=0.6, # Reserve some memory (although this is eval only process)
            trust_remote_code=True,
            max_lora_rank=args.lora_rank
        )
        lora_req = LoRARequest("sft_adapter", 1, args.model_path)
        print(f"Applying LoRA adapter from: {args.model_path}")
    else:
        print(f"Loading Full Model: {args.model_path}")
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            trust_remote_code=True
        )
        lora_req = None

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True 
    )

    print("Generating responses...")
    outputs = llm.generate(questions, sampling_params, lora_request=lora_req)
    
    print("Grading...")
    total_reward = 0
    count = 0
    
    for output, gt in zip(outputs, ground_truths):
        generated_text = output.outputs[0].text
        # r1_zero_reward_fn expects the full response potentially including </think> <answer> ...
        # Our prompt ends in <think>, so generated_text starts with reasoning.
        # We need to construct the "response" string that r1_zero_reward_fn expects.
        # r1_zero_reward_fn checks for "</think> <answer>" inside.
        
        # If the model generated "Reasoning... </think> <answer> 123 </answer>", that's perfect.
        # We feed exactly that to the grader.
        
        result = r1_zero_reward_fn(response=generated_text, ground_truth=gt)
        total_reward += result["reward"]
        count += 1
    
    accuracy = total_reward / count if count > 0 else 0.0
    print(f"EVAL_RESULT_ACCURACY:{accuracy}") 
    return accuracy

# =============================================================================
# 2. Training Logic
# =============================================================================

def get_model_and_tokenizer(args, device):
    print(f"Loading Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    if args.use_lora:
        print(f"⚡ Initializing LoRA (Rank: {args.lora_rank})")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
    else:
        print("🔥 Full Fine-Tuning")
        model.gradient_checkpointing_enable()

    return model, tokenizer

def save_checkpoint_safe(model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

def trigger_eval_subprocess(args, current_model_path, step_info):
    """
    Triggers the same script in --eval_only mode.
    Handles memory offloading if single card.
    """
    eval_model_path = current_model_path
    
    # Construct command
    cmd = [
        sys.executable, __file__,
        "--eval_only",
        "--model_path", eval_model_path,
        "--test_data", args.test_data,
        "--prompt_template_path", args.prompt_template_path,
        "--eval_device", args.eval_device,
        "--lora_rank", str(args.lora_rank)
    ]
    
    if args.use_lora:
        cmd.append("--use_lora")
        # If using LoRA, we need to pass the ORIGINAL base model path to the eval process
        # because current_model_path only contains the adapter
        cmd.extend(["--base_model_path", args.model_path]) # args.model_path is the base model in training mode
    
    if args.max_eval_samples:
        cmd.extend(["--max_eval_samples", str(args.max_eval_samples)])

    # Manage Environment for Device
    env = os.environ.copy()
    
    # Single Card Logic: If train string and eval device string overlap, assume single card
    is_single_card = (args.train_device == args.eval_device) or (args.single_card_mode)
    
    if is_single_card:
        # We rely on main loop to have offloaded the model before calling this
        pass 
    elif args.dual_card_mode:
        # Dual card: Assuming train= cuda:0, eval= cuda:1 (or specified in args)
        # We just run.
        pass

    print(f"\n[Eval] Launching vLLM evaluation for {step_info}...")
    try:
        # We capture stdout to find the accuracy
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Eval] Usage Error: {result.stderr}")
            return None
        
        # Parse Output for "EVAL_RESULT_ACCURACY:..."
        acc = 0.0
        for line in result.stdout.splitlines():
            if "EVAL_RESULT_ACCURACY:" in line:
                acc = float(line.split(":")[1])
                print(f"[Eval] {step_info} Accuracy: {acc}")
                return acc
        
        print(f"[Eval] Could not parse accuracy. Stdout:\n{result.stdout}")
        return None
        
    except Exception as e:
        print(f"[Eval] Exception: {e}")
        return None

def main(args):
    # Setup WandB if not eval mode
    if args.wandb_project and not args.eval_only and wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    set_seed(args.seed)
    
    # Device Setup
    device = torch.device(args.train_device)
    
    # Model & Tokenizer
    model, tokenizer = get_model_and_tokenizer(args, device)
    
    # Data
    train_loader = get_dataloaders(args, tokenizer)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs
    eval_interval_steps = int(len(train_loader) * args.eval_freq)
    print(f"Total Steps: {total_steps}, Eval Method: Every {args.eval_freq} Epochs ({eval_interval_steps} steps)")

    global_step = 0
    model.train()
    
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            global_step += 1
            
            # --- Training Step ---
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # Forward & Loss
            log_probs_dict = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = log_probs_dict["log_probs"]
            loss = masked_normalize(policy_log_probs, response_mask, 1.0)
            
            # Backward
            loss.backward()
            
            if global_step % args.gradient_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            if wandb:
                wandb.log({"train/loss": loss.item(), "train/step": global_step, "train/epoch": epoch + (progress_bar.n / len(train_loader))})
            
            progress_bar.set_postfix(loss=loss.item())
            
            # --- Evaluation Check ---
            if global_step % eval_interval_steps == 0:
                print("\nReached checkpoint interval. Preparing for evaluation...")
                
                # 1. Save Temp Model
                temp_dir = os.path.join(args.output_dir, "temp_eval_ckpt")
                save_checkpoint_safe(model, tokenizer, temp_dir)
                
                # 2. Resource Management for Single Card
                is_single_card = (args.train_device == args.eval_device) or args.single_card_mode
                
                if is_single_card:
                    print("Single Card Mode: Unloading model to CPU to free VRAM for vLLM...")
                    model.to("cpu")
                    torch.cuda.empty_cache()
                
                # 3. Run Eval
                acc = trigger_eval_subprocess(args, temp_dir, f"Step-{global_step}")
                
                if wandb and acc is not None:
                    wandb.log({"eval/accuracy": acc, "eval/step": global_step})
                
                # 4. Restore Resources
                if is_single_card:
                    print("Restoring model to GPU...")
                    model.to(device)
                    # Note: Optimizer state is on CPU/GPU depending on implementation, 
                    # standard AdamW keeps states on same device as params usually. 
                    # If we moved model to CPU, optimizer params might be on CPU now?
                    # The optimizer hooks into params. If params moved, optimizer might need care or simple .step() handles it?
                    # Actually standard PyTorch optimizer.step() might fail if internal state tensors and param tensors are on different devices.
                    # Moving model.to('cpu') moves the parameters.
                    # The optimizer state (momentum, etc) typically stays where it was instantiated unless manually moved.
                    # This is a known complexity.
                    # For safety in this script, we will iterate optimizer state and move it.
                    pass # We will rely on PyTorch knowing what to do, or re-move optimizer. 
                    # Fix: Move optimizer state manually.
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                
                model.train()

    # Final Save (Optional, defaulted to False as requested, but option exists)
    if args.save_model:
        final_dir = os.path.join(args.output_dir, f"{get_time_str()}_final")
        save_checkpoint_safe(model, tokenizer, final_dir)
        print(f"Model saved to {final_dir}")
    
    if wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Mode
    parser.add_argument("--eval_only", action="store_true", help="Run in Evaluation Mode only (vLLM)")
    
    # Paths
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-Math-1.5B", help="Path to model or base model")
    parser.add_argument("--base_model_path", type=str, default=None, help="Base Model Path (for LoRA eval)")
    parser.add_argument("--train_data", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--test_data", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_run")
    parser.add_argument("--prompt_template_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    
    # Training Params
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_acc_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=16)
    
    # Eval & Hardware Params
    parser.add_argument("--n_samples", type=int, default=None, help="Shortcut for setting both train and eval limits")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_freq", type=float, default=0.5, help="Epoch frequency for evaluation")
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--eval_device", type=str, default="cuda:0")
    parser.add_argument("--single_card_mode", action="store_true", help="Force unload model for eval")
    parser.add_argument("--dual_card_mode", action="store_true", help="Treat eval as separate card")
    
    # Saving
    parser.add_argument("--save_model", action="store_true", help="Save final model")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="copilot-sft-gsm8k")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug mode: single card, few samples, 1 epoch")

    args = parser.parse_args()
    
    if args.debug:
        print("🔧 Debug Mode Enabled")
        args.single_card_mode = True
        if args.train_device != args.eval_device:
            print(f"Overriding eval_device {args.eval_device} to match train_device {args.train_device} for debug mode")
            args.eval_device = args.train_device
            
        if args.max_train_samples is None and args.n_samples is None:
            args.max_train_samples = 20
        if args.max_eval_samples is None and args.n_samples is None:
            args.max_eval_samples = 10
        if args.epochs == 3: # Only override if default
            args.epochs = 1
        # Set a frequent eval for debug to test that path
        args.eval_freq = 0.5 

    if args.n_samples:
        args.max_train_samples = args.n_samples
        args.max_eval_samples = args.n_samples

    if args.eval_only:
        run_evaluation_mode(args)
    else:
        main(args)
