# 基本库
import os
import argparse
import random
import json
from datetime import datetime

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
# 1. 配置与参数解析
# =============================================================================
def parse_args():
    # 使用 argparse.ArgumentParser 设定可用的命令行参数
    parser = argparse.ArgumentParser(description="SFT Training Script for Qwen-1.5B")
    
    # 路径配置：模型，数据，存档
    parser.add_argument("--model_path", type=str,   default="models/Qwen2.5-Math-1.5B", help="模型名称或本地路径")
    parser.add_argument("--train_data", type=str,   default="data/gsm8k/train.jsonl",   help="训练数据路径")
    parser.add_argument("--test_data",  type=str,   default="data/gsm8k/test.jsonl",    help="验证数据路径")
    parser.add_argument("--output_dir", type=str,   default="checkpoints/sft_run",      help="模型保存目录")

    # Wandb 记录
    parser.add_argument("--wandb_project",      type=str, default="qwen-math 1.5b sft", help="Wandb 项目名")
    parser.add_argument("--wandb_run_name",     type=str, default=None,                 help="Wandb Run 名")

    # 训练超参数：数据投放（epoch，batch），序列长度，学习率，lora（rank）
    parser.add_argument("--seed",               type=int,   default=42,     help="随机种子")
    parser.add_argument("--epochs",             type=int,   default=1,      help="训练轮数")
    parser.add_argument("--micro_batch_size",   type=int,   default=1,      help="单次前向传播的样本数(当显存受限时应设置为较小的值)")
    parser.add_argument("--gradient_acc_steps", type=int,   default=16,     help="梯度累积步数")
    parser.add_argument("--max_seq_length",     type=int,   default=1024,   help="序列最大长度，防止显存溢出")
    parser.add_argument("--lr",                 type=float, default=1e-5,   help="学习率")
    
    parser.add_argument("--use_lora",           action="store_true",        help="是否使用 LoRA 微调")
    parser.add_argument("--lora_rank",          type=int,   default=16,     help="LoRA 秩")
    
    args = parser.parse_args()
    return args

def confirm_args(args):
    print("="*30 + " 参数配置 " + "="*30)
    print(json.dumps(vars(args), indent=4, ensure_ascii=False))
    print("="*70)
    while True:
        try:
            choice = input("\n>>> 确认上述参数并运行? [y/n]: ").strip().lower()
            if choice == 'y':
                print("参数确认，开始运行...")
                break
            elif choice == 'n':
                print("已取消运行")
                exit(0)
            else:
                print("输入无效，请输入 'y' 确认或 'n' 取消")
        except KeyboardInterrupt:
            print("\n强制退出")
            exit(0)

def get_time_str():
    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    return time_str

def set_seed(seed):
    """把random、np.random、torch的随机函数的种子都设置成指定的同一个"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =============================================================================
# 2. 获取 model 与 tokenizer
# =============================================================================
def get_model_and_tokenizer(args, device):
    """
    从指定名称或路径加载右 padding 分词器，再加载 bf16 模型到指定 device
    
    同时可按参数选择是否使用 LoRA
    """
    print(f"正在加载分词器: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    

    # 显式设置右填充
    tokenizer.padding_side = "right"
    # # 确保 pad_token 存在
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
        
    print(f"正在加载模型: {args.model_path}")
    # 使用 bf16 ，并将模型加载到 device
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    # 可选：LoRA，本地测试最好使用
    if args.use_lora:
        print("⚡ 当前使用 LoRA ")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=2*args.lora_rank,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters() # 打印一下到底训了多少参数
    else:
        print("🔥 当前全量微调，未使用 LoRA")
        # 开启梯度检查点以节省显存
        model.gradient_checkpointing_enable()

    return model, tokenizer

# =============================================================================
# 3. 制作 Dataset 与 DataLoader（注意：由于使用梯度累积，加载器加载的是 micro batch ）
# =============================================================================
class SFTDataset(Dataset):
    """本项目的Dataset子类，封装list形式的原始sft数据"""
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

    # 自定义 Collate Function：
    # 将从原始数据中采集的一个 list 经过指定的处理方式，转化为能够直接输入模型的 tensor
    def collate_fn(batch):
        # batch 形如 list[dict[str,str]]，其中 dict 形如 {"question":"xxx", "answer":"yyy"}
        prompts = [item["question"] for item in batch]
        responses = [item["answer"] for item in batch]
        
        # 使用预先写好的 tokenize_prompt_and_output
        # 将问答对经过分词和右侧 padding ，处理为 input_ids labels response_mask 这三个 tensor
        encoded_batch = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer
        )
        
#         # 截断超出设定的 max_seq_length 的部分
#         max_len = args.max_seq_length
#         if encoded_batch["input_ids"].shape[1] > max_len:
#             encoded_batch["input_ids"] = encoded_batch["input_ids"][:, :max_len]
#             encoded_batch["labels"] = encoded_batch["labels"][:, :max_len]
#             encoded_batch["response_mask"] = encoded_batch["response_mask"][:, :max_len]
        
        return encoded_batch

    # 创建 DataLoader
    train_loader = DataLoader(
        SFTDataset(train_data), 
        batch_size=args.micro_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 验证集也有自己的 DataLoader ，区别是可以不需要 shuffle
    # 或许 batch size 可以再大一点？目前没有做区分
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
# def evaluate(model, val_loader, device):
#     """验证集评估函数：只计算 Loss，不进行生成"""
#     model.eval()
#     total_loss = 0
#     total_steps = 0
    
#     print("正在进行验证集评估...")
#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc="Evaluating"):
#             # 1. 数据上卡
#             input_ids = batch["input_ids"].to(device)
#             labels = batch["labels"].to(device)
#             response_mask = batch["response_mask"].to(device)
            
#             # 2. 计算 Logits
#             # 注意：get_response_log_probs 内部调用了 model(input_ids)
#             log_probs_dict = get_response_log_probs(model, input_ids, labels)
#             policy_log_probs = log_probs_dict["log_probs"]
            
#             # 3. 计算 Loss
#             # 这里我们不调用 sft_microbatch_train_step，因为它包含 backward
#             # 我们直接调用 masked_normalize 计算纯 loss
#             # 注意：验证时不涉及梯度累积，所以不需要除以 accumulation steps
#             # 直接计算这个 batch 的平均 loss
#             loss = -masked_normalize(
#                 policy_log_probs,
#                 response_mask,
#                 normalize_constant=1.0, # 默认为 1
#                 dim=None
#             ) / input_ids.shape[0] # 除以 batch_size 得到平均 loss
            
#             total_loss += loss.item()
#             total_steps += 1
            
#     avg_loss = total_loss / total_steps
#     model.train() # 切回训练模式
#     return avg_loss

def save_checkpoint(model, tokenizer, args, step_or_epoch):
    """保存模型：兼容 LoRA 和全量"""
    save_path = os.path.join(args.output_dir, args.start_time, f"{args.save_time}_checkpoint-{step_or_epoch}")
    print(f"正在保存模型到: {save_path}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 保存 tokenizer
    tokenizer.save_pretrained(save_path)
    
    # 保存模型（无论是否使用 LoRA）
    model.save_pretrained(save_path)
    print("保存完成。")

# =============================================================================
# 5. 构建完整流程，具体制定训练循环
# =============================================================================
def main(args):
    # 1. 初始化环境：种子，设备，wandb
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
    
    # 2. 准备组件：模型，分词器，数据加载器，优化器
    model, tokenizer = get_model_and_tokenizer(args, device)
    train_loader, val_loader = get_dataloaders(args, tokenizer)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad()   # 清空梯度

    # 3. 训练循环
    big_step = 0    # 记录逻辑 batch 的步数
    big_loss = 0.0  # 记录逻辑 batch 的 loss （ micro_batch 之间无优化器步进，持续观察 micro_loss 不合适，应当累加）
    
    print(f"开始训练，总轮数: {args.epochs}")
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for micro_step, micro_batch in enumerate(progress_bar):
            # batch 是从 train_loader 中取出的，遍历它即可获得每个 micro_batch 要用到的数据

            # --- 搬运数据 ---
            input_ids = micro_batch["input_ids"].to(device)
            labels = micro_batch["labels"].to(device)
            response_mask = micro_batch["response_mask"].to(device)
            
            # --- 前向传播，获得对数概率 ---
            log_probs_dict = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = log_probs_dict["log_probs"]
            
            # --- 计算 loss，反向传播 ---
            micro_loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_acc_steps
            )
            big_loss += micro_loss.item()
            
            # --- 梯度累积达到一定步数，逻辑 batch 结束 ---
            # 每个 batch 运行完的工作：报告 loss，优化器移动一步，清空梯度，记录日志信息等
            if (micro_step + 1) % args.gradient_acc_steps == 0:
                # 梯度裁剪 (防止梯度爆炸)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # 记录到 Wandb
                wandb.log({
                    "train/loss": big_loss, 
                    "train/epoch": epoch + (micro_step + 1) / len(train_loader),
                    "big_step": big_step
                })
                
                # 更新进度条
                progress_bar.set_postfix({"loss": big_loss})
                
                # 重置累积器
                big_loss = 0.0
                big_step += 1
                
        # --- F. 每个 Epoch 结束后验证与保存 ---
        # val_loss = evaluate(model, val_loader, device)
        # print(f"Epoch {epoch+1} 验证集 Loss: {val_loss:.4f}")
        # wandb.log({"eval/loss": val_loss, "eval/epoch": epoch + 1})

        save_time_str = get_time_str()
        args.save_time = save_time_str
        
        save_checkpoint(model, tokenizer, args, f"epoch-{epoch+1}")

    print("训练结束！")
    wandb.finish()

# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    start_time_str = get_time_str()
    args = parse_args()
    args.start_time = start_time_str
    confirm_args(args)

    main(args)