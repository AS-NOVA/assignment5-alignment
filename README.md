# CS336 Spring 2025 Assignment 5: Alignment

## Setup


> 曾经的成功经历
>```
> pip install --no-build-isolation -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
> 由于在flash-attn卡住，所以
> ```
> pip install flash-attn --no-build-isolation
> ```
> 之后再重新执行第一条命令

uv换源后，`uv sync`可以快速解决其他部分的安装，但flash-attn仍然有问题。

作业本身的指示：

``` sh
uv sync --no-install-package flash-attn
uv sync
```

但是还是会在flash-attn处卡住，原因不明