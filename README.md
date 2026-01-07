# CS336 Spring 2025 Assignment 5: Alignment

## 环境配置

> autodl学术代理的启动，如果需要
> 
> `source /etc/network_turbo`
> 
> 退出加速：
> 
> `unset http_proxy && unset https_proxy`

> 曾经的成功经历
> 
> ```bash
> pip install --no-build-isolation -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```
> 
> 由于在flash-attn卡住，所以
> 
> ```bash
> pip install flash-attn --no-build-isolation
> ```
> 
> 之后再重新执行第一条命令

uv换源后，`uv sync`可以快速解决其他部分的安装，但flash-attn仍然有问题。

作业本身的指示：

```bash
uv sync --no-install-package flash-attn
uv sync
```

然而有两个地方有问题

1. alpaca-eval需要走github，autodl环境下需要启动学术代理，然而这会和uv换源（这里是阿里源）冲突。

为了解决alpaca-eval，可以临时将uv源改回来：

`uv sync --no-install-package flash-attn --index-url https://pypi.org/simple`



2. flash-attn本地编译非常的慢

为此，应该考虑从github的release中直接观察需要的版本whl名称，直接通过对应的whl链接安装

本项目中默认使用的版本是`2.7.4.post1`，我们在github主页`https://github.com/Dao-AILab/flash-attention/releases`上寻找它

每个flash_attn对应的链接应该类似这样：

`https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`

可以将其中的包名分段：

- `flash_attn-2.7.4.post1`：flash-attn版本名
- `+cu12torch2.5`：对应cuda/torch版本
- `cxx11abiFALSE`：可以通过运行`python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)"`查看应该是true还是false
- `-cp312-cp312`：python版本
- `-linux_x86_64.whl`：linux系统版本

知道正确的包名后，获得对应链接，直接`pip install <链接>即可`



激活uv环境：`source .venv/bin/activate`





## 模型下载

本地下载较为麻烦，考虑先直接用hf库的缓存功能。

首先设置镜像站，将`HF_ENDPOINT`设置为镜像站，并且写到`~/.bashrc`中：

```bash
echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
```

另外，在autodl上，如果担心hf把模型缓存到系统盘，导致空间不足，可以设置将模型默认缓存至数据盘：

```bash
echo 'export HF_HOME="/root/autodl-tmp/hf_cache"' >> ~/.bashrc
```

无论往`~/.bashrc`中写了什么，记得刷新配置

```bash
source ~/.bashrc
```


## 运行脚本

快速测试

```bash
python scripts/my_sft_train.py \
    --model_path "Qwen/Qwen2.5-Math-1.5B" \
    --use_lora \
    --micro_batch_size 1 \
    --gradient_acc_steps 8 \
    --lr 2e-4 \
    --epochs 1 \
    --wandb_project "sft-local-test"
```

全量微调

```bash
python scripts/my_sft_train.py \
    --model_path "Qwen/Qwen2.5-Math-1.5B" \
    --output_dir "checkpoints/sft_full"\
    --micro_batch_size 1 \
    --gradient_acc_steps 16 \
    --lr 1e-5 \
    --epochs 1 \
    --wandb_project "sft-full-finetune"
```


