#!/usr/bin/env python3

# 基本库
import os
import argparse
import random

# 数值计算与深度学习
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

# 大模型
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 自定义库
import sys
# 将当前文件（.py）的所在文件夹（scripts）的所在文件夹（项目根目录）加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cs336_alignment.my_sft_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    masked_normalize
)
from cs336_alignment.my_sft_toolfunc import (
    read_jsonl,
    get_qa_list
)

# =============================================================================
# 1. 配置与参数解析 (Configuration)
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training Script for Qwen-1.5B")
    
    # 路径配置
    # - 模型：模型路径 model_path
    # - 数据：数据文件 train_data, test_data
    # - 存档：存档路径 output_dir
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-Math-1.5B", help="本地模型路径")
    parser.add_argument("--train_data", type=str, default="data/gsm8k/train.jsonl", help="训练数据路径")
    parser.add_argument("--test_data", type=str, default="data/gsm8k/test.jsonl", help="验证数据路径")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_run", help="模型保存目录")
    
    # 训练超参数
    # - 随机：种子 seed
    # - 数据投放：
    #       数据集遍历次数 epochs，
    #       理论 batch 大小 = micro_batch_size * gradient_accumulation_steps，
    #       序列长度 max_seq_length
    # - 优化器：
    #       学习率 lr
    #       LoRA 开关 use_lora
    #       LoRA rank lora_rank
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="单次前向传播的样本数(受显存限制)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="序列最大长度，防止OOM")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    
    parser.add_argument("--use_lora", action="store_true", help="是否使用 LoRA 微调")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA Rank")
    
    # Wandb 记录
    # - 名称：
    #       项目名 wandb_project
    #       运行名 wandb_run_name
    parser.add_argument("--wandb_project", type=str, default="cs336-sft", help="Wandb 项目名")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb Run 名")

    args = parser.parse_args()
    return args

def set_seed(seed):
    """固定随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =============================================================================
# 2. 模型与分词器工厂 (Model Factory)
# =============================================================================
def get_model_and_tokenizer(args, device):
    print(f"正在加载分词器: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    

    # 显式设置 padding_side 为 right，虽然你的代码逻辑支持右填充，但明确设置更安全
    tokenizer.padding_side = "right"
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"正在加载模型: {args.model_path}")
    # 使用 bfloat16 以节省显存并保持精度 (40系显卡支持 bf16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    # 显存优化策略：LoRA
    if args.use_lora:
        print("⚡ 启用 LoRA 模式...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            # 针对 Qwen/Llama 的常见线性层
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters() # 打印一下到底训了多少参数
    else:
        print("🔥 启用全量微调模式 (注意显存)...")
        # 如果全量微调，建议开启梯度检查点以节省显存
        model.gradient_checkpointing_enable()

    return model, tokenizer

# =============================================================================
# 3. 数据管道工厂 (Data Factory)
# =============================================================================
class SFTDataset(Dataset):
    """简单的 Dataset 包装器，配合 DataLoader 使用"""
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloaders(args, tokenizer):
    print("正在读取数据...")
    # 使用你写的 read_jsonl 读取数据
    train_data = get_qa_list(read_jsonl(args.train_data))
    val_data = get_qa_list(read_jsonl(args.test_data))
    
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")

    # 定义 Collate Function：这是连接数据和模型的关键
    # 它负责把一个 batch 的 raw data 转换成 tensor
    def collate_fn(batch):
        # batch 是一个 list of dict: [{"question": "...", "answer": "..."}, ...]
        prompts = [item["question"] for item in batch]
        responses = [item["answer"] for item in batch]
        
        # 调用你写好的 tokenize_prompt_and_output
        # 注意：这里使用的是 Right Padding，这对于 SFT 训练是没问题的
        encoded_batch = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer
        )
        
        # 简单的长度截断保护 (虽然你之前说 GSM8K 不长，但加上更稳健)
        # 如果超过 max_seq_length，进行切片
        max_len = args.max_seq_length
        if encoded_batch["input_ids"].shape[1] > max_len:
            encoded_batch["input_ids"] = encoded_batch["input_ids"][:, :max_len]
            encoded_batch["labels"] = encoded_batch["labels"][:, :max_len]
            encoded_batch["response_mask"] = encoded_batch["response_mask"][:, :max_len]
            
        return encoded_batch

    # 创建 DataLoader
    train_loader = DataLoader(
        SFTDataset(train_data), 
        batch_size=args.micro_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 验证集通常不需要 shuffle，batch size 可以稍微大一点（如果不做反向传播）
    val_loader = DataLoader(
        SFTDataset(val_data), 
        batch_size=args.micro_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

# =============================================================================
# 4. 验证与保存逻辑 (Eval & Save)
# =============================================================================
def evaluate(model, val_loader, device):
    """验证集评估函数：只计算 Loss，不进行生成"""
    model.eval()
    total_loss = 0
    total_steps = 0
    
    print("正在进行验证集评估...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # 1. 数据上卡
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # 2. 计算 Logits
            # 注意：get_response_log_probs 内部调用了 model(input_ids)
            log_probs_dict = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = log_probs_dict["log_probs"]
            
            # 3. 计算 Loss
            # 这里我们不调用 sft_microbatch_train_step，因为它包含 backward
            # 我们直接调用 masked_normalize 计算纯 loss
            # 注意：验证时不涉及梯度累积，所以不需要除以 accumulation steps
            # 直接计算这个 batch 的平均 loss
            loss = -masked_normalize(
                policy_log_probs,
                response_mask,
                normalize_constant=1.0, # 默认为 1
                dim=None
            ) / input_ids.shape[0] # 除以 batch_size 得到平均 loss
            
            total_loss += loss.item()
            total_steps += 1
            
    avg_loss = total_loss / total_steps
    model.train() # 切回训练模式
    return avg_loss

def save_checkpoint(model, tokenizer, args, step_or_epoch):
    """保存模型：兼容 LoRA 和全量"""
    save_path = os.path.join(args.output_dir, f"checkpoint-{step_or_epoch}")
    print(f"正在保存模型到: {save_path}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 保存 tokenizer
    tokenizer.save_pretrained(save_path)
    
    # 保存模型
    if args.use_lora:
        # LoRA 模式下，save_pretrained 只会保存 adapter 权重 (很小)
        model.save_pretrained(save_path)
    else:
        # 全量模式下，保存完整权重
        model.save_pretrained(save_path)
    print("保存完成。")

# =============================================================================
# 5. 主训练循环 (Main Logic)
# =============================================================================
def main(args):
    # 1. 初始化环境
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化 Wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
    
    # 2. 准备组件
    model, tokenizer = get_model_and_tokenizer(args, device)
    train_loader, val_loader = get_dataloaders(args, tokenizer)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 简单的学习率调度器 (Linear Warmup + Decay 可以后续再加，先用简单的)
    # scheduler = ... 
    
    # 3. 训练循环
    global_step = 0
    optimizer.zero_grad()
    accumulated_loss = 0.0 # 用于记录一个大 Batch 的 loss (用于 Wandb)
    
    print(f"开始训练，总轮数: {args.epochs}")
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # --- A. 数据上卡 ---
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # --- B. 前向传播 (计算 Logits) ---
            # 使用你写的函数：获得 p(label|input)
            log_probs_dict = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = log_probs_dict["log_probs"]
            
            # --- C. 反向传播 (Microbatch Step) ---
            # 使用你写的函数：计算 Loss 并 Backward
            # 注意：sft_microbatch_train_step 内部已经除以了 gradient_accumulation_steps
            # 并执行了 loss.backward()
            loss_micro, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
            
            # --- D. Loss 记录逻辑 ---
            # 因为 loss_micro 已经是 (L_total / Acc / B)
            # 我们直接累加它：Sum(loss_micro) = Sum(L_total / B) / Acc = Avg_Loss_Global
            # 所以这里不需要再除以 Acc，直接加即可。
            accumulated_loss += loss_micro.item()
            
            # --- E. 梯度更新 (Gradient Update) ---
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 梯度裁剪 (防止梯度爆炸)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # 记录到 Wandb
                wandb.log({
                    "train/loss": accumulated_loss, 
                    "train/epoch": epoch + (step + 1) / len(train_loader),
                    "global_step": global_step
                })
                
                # 更新进度条
                progress_bar.set_postfix({"loss": accumulated_loss})
                
                # 重置累积器
                accumulated_loss = 0.0
                global_step += 1
                
        # --- F. 每个 Epoch 结束后验证与保存 ---
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} 验证集 Loss: {val_loss:.4f}")
        wandb.log({"eval/loss": val_loss, "eval/epoch": epoch + 1})
        
        save_checkpoint(model, tokenizer, args, f"epoch-{epoch+1}")

    print("训练结束！")
    wandb.finish()

# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)