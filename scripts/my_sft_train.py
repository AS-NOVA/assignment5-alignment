import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm
import wandb

# 将父目录加入路径，确保能导入 cs336_alignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_alignment.my_sft_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step
)
from cs336_alignment.my_sft_toolfunc import read_jsonl, get_qa_list

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    # ================= 1. 配置与初始化 =================
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化 wandb
    wandb.init(
        project="sft-gsm8k-1.5b",
        config=vars(args),
        mode="online" if not args.debug else "disabled" # debug 模式下不上传日志
    )

    # ================= 2. 数据管道 =================
    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Qwen 的 pad_token 通常是 eos_token，或者是 none，需要手动处理一下以防万一
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 读取数据 (使用你写的工具函数)
    raw_data = get_qa_list(read_jsonl(args.data_path))
    
    if args.debug:
        print("Debug mode: utilizing only first 16 examples.")
        raw_data = raw_data[:16]

    print(f"Total examples: {len(raw_data)}")

    # 自定义 Collate Function：把一个 Batch 的原始数据转换成 Tensor
    # 这里我们利用你写的 tokenize_prompt_and_output
    def collate_fn(batch_list):
        # batch_list 是一个 list of dict, e.g. [{"question":..., "answer":...}, ...]
        questions = [item["question"] for item in batch_list]
        answers = [item["answer"] for item in batch_list]
        
        # 调用你的分词函数
        tokenized_data = tokenize_prompt_and_output(
            prompt_strs=questions,
            output_strs=answers,
            tokenizer=tokenizer
        )
        return tokenized_data

    # 构建 DataLoader
    train_loader = DataLoader(
        raw_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True # 丢弃最后不足一个 batch 的数据，防止形状变化导致显存碎片
    )

    # ================= 3. 模型准备 =================
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16, # 4060 支持 bf16，这很重要，省显存
        use_cache=False             # 训练时不需要 KV Cache
    ).to(device)
    
    # 开启梯度检查点 (Gradient Checkpointing) - 4060 只有 8G 显存，这几乎是必须的
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ================= 4. 训练主循环 =================
    model.train()
    global_step = 0
    total_loss = 0.0
    
    # 计算总步数用于进度条
    total_steps = len(train_loader) * args.epochs
    progress_bar = tqdm(total=total_steps, desc="Training")

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            # batch 是 collate_fn 返回的字典
            
            # 1. 数据移到 GPU
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # 2. 前向传播：获取 Log Probabilities
            # 利用你写的函数 get_response_log_probs
            # 注意：这个函数内部已经包含了 model(input_ids) 的调用
            log_probs_dict = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True # 顺便记录一下熵，作为监控指标
            )
            policy_log_probs = log_probs_dict["log_probs"]
            token_entropy = log_probs_dict["token_entropy"]

            # 3. 计算 Loss (SFT 微调的核心)
            # 利用你写的 sft_microbatch_train_step
            # 注意：这个函数内部已经包含了 loss.backward()
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.grad_accum_steps
            )

            # 记录累积 loss 用于显示
            total_loss += loss.item() * args.grad_accum_steps 

            # 4. 优化器更新 (处理梯度累积)
            if (step + 1) % args.grad_accum_steps == 0:
                # 梯度裁剪 (防止梯度爆炸)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # 5. 记录日志
                avg_entropy = (token_entropy * response_mask).sum() / response_mask.sum()
                
                wandb.log({
                    "train/loss": total_loss / args.grad_accum_steps,
                    "train/entropy": avg_entropy.item(),
                    "train/epoch": epoch + (step + 1) / len(train_loader),
                    "train/step": global_step
                })
                
                progress_bar.set_postfix(loss=total_loss / args.grad_accum_steps)
                total_loss = 0.0
                global_step += 1
                progress_bar.update(1)

    progress_bar.close()

    # ================= 5. 保存模型 =================
    print(f"Saving model to {args.save_path}...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training Script")
    
    # 路径参数
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--data_path", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--save_path", type=str, default="checkpoints/sft_final")
    
    # 训练超参
    parser.add_argument("--batch_size", type=int, default=1, help="Micro batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save memory")
    
    # 调试参数
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer steps, no wandb upload)")

    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    main(args)