# CS336 Spring 2025 Assignment 5: Alignment

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
``` sh
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.

To connect your implementation to the tests, complete the functions in [./tests/adapters.py](./tests/adapters.py).

## autodl 环境配置

本次尝试的是使用autodl自带的pytorch2.5.1镜像。

### autodl 免密登录

本机powershell输入`ssh-keygen -t rsa`，一路回车，到`C:\Users\NOVA\.ssh`找到两个rsa文件，将.pub后缀的公钥内容复制，粘贴到`https://www.autodl.com/console/instance`界面“设置密钥登录”内

### autodl 联网

启动autodl自带的网络加速脚本：

`source /etc/network_turbo`

（退出网络加速：`unset http_proxy && unset https_proxy`）

下载clash for autodl

`git clone https://github.com/VocabVictor/clash-for-AutoDL.git`

进入后打开.env文件

```sh
cd clash-for-AutoDL
cp .env.example .env
vim .env
```

按i进入编辑模式，粘贴clash_url

ESC退出编辑模式，输入 :w 按enter保存，输入 :q 按enter退出

