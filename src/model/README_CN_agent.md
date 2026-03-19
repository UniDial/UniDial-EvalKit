<h1 align="center">
  <img src="assets/logo.jpg" alt="UDE Logo" width="100" height="100" align="absmiddle">
  UniDial-EvalKit
</h1>


## 🛠️ 安装方法

1. 克隆本项目代码：
   ```bash
   git clone https://github.com/JiaQiSJTU/UniDial-EvalKit.git
   cd UniDial-EvalKit
   ```

2. 创建并激活 conda 环境：
   ```bash
   conda create -n uniconv python=3.10
   conda activate uniconv
   ```

3. 安装依赖环境：
   ```bash
   pip install -r requirements.txt
   ```


## 🤖 Agent 评测部署与使用说明

本评测框架内置了对多种多轮对话 Agent（**HippoRAG**, **MemoryOS**, **AMem**）的适配支持。
**注意：** 因为这些 Agent 本身是独立的代码工程，本框架为了最大程度解耦，需要你将对应的核心源码手动放置在指定位置，再由本框架中的适配器进行调用。

---

### 1. 🦛 HippoRAG Agent
HippoRAG 是基于图谱优化的检索增强生成 Agent。

#### 1.1 下载与目录配置
你需要从官方 GitHub 仓库下载 HippoRAG 的源代码，并将其中提供核心功能的 `hipporag` 目录提取出来。

1. 在任意临时目录克隆 HippoRAG 官方仓库：
   ```bash
   git clone https://github.com/OSU-NLP-Group/HippoRAG.git
   ```
2. 将克隆下来的仓库中的 `src/hipporag` 文件夹（整个文件夹）复制到本项目的 **`src/model/`** 目录下。放置后，目录结构如下：
   ```text
   UniDial-EvalKit/
   ├── src/
   │   ├── model/
   │   │   ├── hipporag_agent.py   # 我们封装的 Agent 接口
   │   │   └── hipporag/           # <--- 放置在这里 (从外部仓库复制来的)
   │   │       ├── __init__.py
   │   │       ├── HippoRAG.py
   │   │       └── ...
   ```

#### 1.2 环境依赖安装
进入你刚刚拉取的 `HippoRAG` 原始仓库目录，或者直接使用它内部的 `requirements.txt` 进行安装：
```bash
conda activate uniconv
# 假设你在临时目录下载了仓库，安装其依赖：
pip install -r /path/to/HippoRAG/requirements.txt
```

#### 1.3 命令行评测
配置完成后，将 `--model_type` 设置为 `hipporag` 即可评测。底座调用的模型通过 `--model_name` 指定。
```bash


PYTHONPATH=. python src/eval_cli.py \
    --dataset personamem \
    --raw_data_dir ./raw_data/PersonaMem \
    --model_type hipporag \
    --model_name "deepseek-v3.2" \
    --base_url "https://api.your-proxy.com/v1" \
    --api_key "your_api_key" \
    --do_generation \
    --do_evaluation \
    --parallel 4
```

---

### 2. 🧠 MemoryOS Agent
MemoryOS 是一个具有短期、中期和长期层级记忆的 Agent 框架。

#### 2.1 下载与目录配置
1. 从官方 GitHub 仓库克隆 `MemoryOS` 项目：
   ```bash
   git clone https://github.com/BAI-LAB/MemoryOS.git
   ```
2. 将克隆下来的项目中 `memoryos-pypi` 文件夹内的内容（包含 `memoryos.py`, `retriever.py`, `utils.py` 等），**重命名为 `memoryos`**，并放置到本项目的 **`src/model/`** 目录下：
   ```text
   UniDial-EvalKit/
   ├── src/
   │   ├── model/
   │   │   ├── memoryos_agent.py   # 框架适配器
   │   │   └── memoryos/           # <--- 从官方仓库 memoryos-pypi 复制并重命名而来
   │   │       ├── __init__.py
   │   │       ├── memoryos.py
   │   │       ├── utils.py        # (务必确保 utils.py 存在，否则会报 Import 错误)
   │   │       └── ...
   ```

#### 2.2 环境依赖安装
在项目根目录下安装 `src/model/memoryos` 的专属依赖：
```bash
conda activate uniconv
pip install -r src/model/memoryos/requirements.txt
```

#### 2.3 命令行评测
配置完成后，将 `--model_name` 设置为你要用的大模型名，`--model_type` 设置为 `memoryos`。
```bash
PYTHONPATH=. python src/eval_cli.py \
    --dataset personamem \
    --raw_data_dir ./raw_data/PersonaMem \
    --model_type memoryos \
    --model_name "deepseek-chat" \
    --base_url "https://api.deepseek.com" \
    --api_key "your-api-key" \
    --do_generation \
    --do_evaluation \
    --parallel 4
```

---

### 3. 🤖 AMem Agent
AgenticMemory (AMem) 是一个轻量级代理化记忆管理系统。

#### 3.1 下载与目录配置
你需要获取 A-mem 项目的原始代码并放置到正确的位置。

1. 从对应的 GitHub 仓库（或源码压缩包中）获取 `A-mem` 核心代码。
   ```bash
   # 假设你通过 Git 克隆了原始项目：
   git clone https://github.com/WujiangXu/A-mem.git
   ```
2. 将获取到的项目中的核心记忆代码包（即包含 `memory_layer.py` 的文件夹，比如 `A-mem-main` 或者 `A-mem` 下的代码），**重命名为 `AgenticMemory`**，并将其放置于本项目的 **`src/model/`** 目录下：
   ```text
   UniDial-EvalKit/
   ├── src/
   │   ├── model/
   │   │   ├── amem_agent.py       # 框架适配器
   │   │   └── AgenticMemory/      # <--- 放置在这里，文件夹必须叫这个名字
   │   │       ├── memory_layer.py # 确保里面包含这个核心逻辑文件
   │   │       └── ...
   ```

#### 3.2 环境依赖安装
AMem 的执行依赖于特定的第三方库，请在项目根目录下安装 `src/model/AgenticMemory` 的依赖：
```bash
conda activate uniconv
pip install -r src/model/AgenticMemory/requirements.txt
```

#### 3.3 命令行评测
```bash
PYTHONPATH=. python src/eval_cli.py \
    --dataset personamem \
    --raw_data_dir ./raw_data/PersonaMem \
    --model_type amem \
    --model_name "deepseek-v3.2" \
    --base_url "https://api.deepseek.com" \
    --api_key "your-api-key" \
    --do_generation \
    --do_evaluation \
    --parallel 4
```

---







