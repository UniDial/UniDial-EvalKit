# UniConv-EvalKit

**UniConv-EvalKit** 是一个专注于 **大模型多轮交互评测** 的统一框架。我们致力于构建覆盖 **记忆、理解、安全、数学、代码** 等多维度的长期交互综合能力评价体系。通过集成多个主流Benchmark进行综合对比，UniConv-EvalKit为大语言模型及 Agent系统的多轮人机交互技术演进提供全方位、标准化的能力画像。

作为一款模块化、可扩展的评测工具，UniConv-EvalKit 将数据加载、模型生成与指标评测解耦，支持断点续传与并行加速，能够高效地对各类开源及闭源模型进行全方位能力评估。

## ✨ 项目亮点

- **模块化架构**：核心组件（`Dataset`, `Model`, `Metric`）完全解耦，易于扩展新的数据集、评测对象或评测指标。
- **高效并行**：支持多线程并行生成与评测，大幅缩短大规模评估时间。
- **断点续传**：内置结果缓存机制，自动跳过已完成的对话或评测条目，避免意外中断导致的重复推理计算。
- **丰富的评测基准**：内置多种前沿评测数据集支持，包括但不限于：
  - **通用能力**：`mt_eval`, `mt_bench_101`, `multi_challenge`
  - **记忆**：`longmemeval`, `locomo`
  - **安全**：`safedialbench`
  - **特定任务**：`mathchat` (数学), `memorycode`(代码), `multi_if`（指令遵循）, `personamem`(个性化)
- **灵活的聚合统计**：支持从 Turn (轮次) 到 Dialog (对话) 再到 Dataset (数据集) 的多级分数聚合策略（Mean/Min/Max）。

## 📊 Leaderboard

以下为使用 UniConv-EvalKit 对 **DeepSeek-V3.2** 进行多维度评测的结果（Judge Model: GPT-4.1）：

|               | 评测指标                       | DeepSeek-V3.2 |
|---------------|------------------------------|---------------|
| LoCoMo        | `f1_score`, `recall`         | 59.25         |
| MathChat      | `llm_judge`, `numeric_match` | 77.87         |
| MemoryCode    | `code_math`                  | 25.40         |
| MT-Bench-101  | `llm_judge`                  | 91.17         |
| PersonaMem    | `exact_match`                | 60.88         |
| MultiIF       | `instruction_following`      |               |
| SafeDialBench | `llm_judge`                  |               |


> ⚠️ 全部评测均以多轮 user-assistant 交互形式进行，评测设定可能与原文存在差异。分数结果按 `agg_turn_stat=mean`, `agg_dialog_stat=min`, `agg_dataset_level=dialog` 进行汇总。

> 🔄 更多模型的评测结果与深入分析将逐步 release，敬请关注！

## 🛠️ 安装方法

1. 克隆本项目代码：
   ```bash
   git clone https://github.com/JiaQiSJTU/UniConv-EvalKit.git
   cd UniConv-EvalKit
   ```

2. 安装依赖环境：
   ```bash
   pip install -r requirements.txt
   ```

## 📦 数据准备

在运行评测前，需要将对应数据集的原始数据文件放置在 `raw_data/` 目录下。各数据集所需文件如下：

| Benchmark | `--dataset` | 能力维度 | 数据源 | 参考文献 |
| :--- | :--- | :--- | :--- | :--- |
| MT-Eval | `mt_eval` | 通用能力 | [🤗 Hugging Face](https://huggingface.co/datasets/wckwan/MT-Eval)(`*.jsonl`) | [2024EMNLP](https://aclanthology.org/2024.emnlp-main.1124.pdf) |
| MT-Bench-101 | `mt_bench_101` | 通用能力 | [:octocat: Github](https://github.com/mtbench101/mt-bench-101)(`*.jsonl`) | [2024ACL](https://aclanthology.org/2024.acl-long.401/) |
| Multi-Challenge | `multi_challenge` | 通用能力 | [:octocat: Github](https://github.com/ekwinox117/multi-challenge)(`*.jsonl`) | [2025ACLfindings](https://aclanthology.org/2025.findings-acl.958/) |
| LongMemEval | `longmemeval` | 记忆 | [🤗 Hugging Face](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)(`*.json`) | [2025ICLR Poster](https://openreview.net/pdf?id=pZiyCaVuti) |
| LoCoMo | `locomo` | 记忆 | [:octocat: Github](https://github.com/snap-research/locomo/tree/main/data)(`*.json`) | [2024ACL](https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2024.acl-long.747.pdf) |
| MathChat | `mathchat` | 数学 | [:octocat: Github](https://github.com/Zhenwen-NLP/MathChat)(`*.jsonl`) | [Arxiv](https://arxiv.org/pdf/2405.19444) |
| MemoryCode | `memorycode` | 代码 | [:octocat: Github](https://github.com/Cohere-Labs-Community/MemoryCode)(`dialogue_*.json`) | [2025ACL](https://arxiv.org/abs/2502.13791) |
| Multi-IF | `multi_if` | 指令遵循 | [🤗 Hugging Face](https://huggingface.co/datasets/facebook/Multi-IF)(`*.csv`) | [2025ACL](https://aclanthology.org/2025.acl-long.1172.pdf) |
| PersonaMem | `personamem` | 个性化 | [🤗 Hugging Face](https://huggingface.co/datasets/bowen-upenn/PersonaMem)(`questions_*.csv`, `shared_contexts_*.jsonl`) | [2025COLM](https://arxiv.org/abs/2504.14225) |
| SafeDialBench | `safedialbench` | 安全 | [🤗 Hugging Face](https://huggingface.co/datasets/HongyeCao/SafeDialBench)(`datasets_*.jsonl`及prompt文件`*.jsonl`) | [2026ICLR](https://arxiv.org/abs/2502.11090) |

将对应文件放入 `raw_data/<目录名>/` 后，通过 `--raw_data_dir` 参数指定路径即可，例如：

```bash
--raw_data_dir "./raw_data/MT-Eval"
```

## 🚀 使用方法

评测流程主要分为 **数据加载**、**模型生成 (Generation)**、**指标评测 (Evaluation)** 和 **结果聚合** 四个阶段。支持 **CLI 命令行** 和 **Python 编程** 两种调用方式（详见 [`eval_pipeline.md`](eval_pipeline.md)）。

### 1. CLI 命令行

入口脚本为 `src/eval_cli.py`，通过命令行参数灵活控制。

**完整流程（生成 + 评测）：**

```bash
PYTHONPATH=. python src/eval_cli.py \
    --dataset mt_eval \
    --raw_data_dir ./raw_data/MT-Eval \
    --model_name gpt-3.5-turbo \
    --judge_model_name gpt-4.1 \
    --base_url xxx \
    --api_key "your-api-key" \
    --do_generation \
    --do_evaluation \
    --parallel 4
```

**仅生成回答：**

```bash
PYTHONPATH=. python src/eval_cli.py \
    --dataset mathchat \
    --raw_data_dir ./raw_data/MathChat \
    --model_name deepseek-chat \
    --base_url xxx \
    --api_key "your-api-key" \
    --do_generation
```

**仅运行评测（需已有生成结果）：**

```bash
PYTHONPATH=. python src/eval_cli.py \
    --dataset mathchat \
    --model_name deepseek-chat \
    --judge_model_name gpt-4.1 \
    --base_url xxx \
    --api_key "your-api-key" \
    --do_evaluation
```

### 2. Python 编程调用

通过 `EvalPipeline` 和 `EvalPipelineConfig`，可在脚本或 Notebook 中灵活调用，支持分阶段精细控制。

```python
from src.eval_pipeline import EvalPipeline, EvalPipelineConfig

cfg = EvalPipelineConfig(
    dataset="mt_eval",
    raw_data_dir="./raw_data/MT-Eval",
    model_name="gpt-3.5-turbo",
    judge_model_name="gpt-4",
    base_url="xxx",
    api_key="your-api-key",
    do_generation=True,
    do_evaluation=True,
    parallel=4,
)

# 一键执行
pipeline = EvalPipeline(cfg)
pipeline.run()

# 或分阶段调用（支持中间自定义处理）
# dialogs   = pipeline.prepare_data()
# generated = pipeline.run_generation(dialogs)
# results   = pipeline.run_evaluation(generated)
# summary   = pipeline.run_aggregation(results)
```

### 3. 使用 vLLM 部署本地模型

对于开源模型，可通过 [vLLM](https://docs.vllm.ai/) 部署 OpenAI 兼容的推理服务，然后直接对接本框架进行评测。

**第一步：启动 vLLM 服务**（参见 `script/vllm_server.sh`）

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model ./Models/Qwen3-8B \
    --served-model-name "Qwen3-8B"
```

服务默认监听 `http://localhost:8000`。

**第二步：运行评测**，通过 `--base_url` 指向本地 vLLM 服务（参见 `script/test_vllm_client.sh`）

```bash
PYTHONPATH=. python src/eval_cli.py \
    --dataset locomo \
    --raw_data_dir ./raw_data/LoCoMo \
    --model_name Qwen3-8B \
    --base_url http://localhost:8000/v1/ \
    --do_generation
```

> 💡 vLLM 提供的是 OpenAI 兼容接口，因此 `--model_type` 保持默认 `openai` 即可，只需将 `--base_url` 指向本地地址，`--model_name` 设为 `--served-model-name` 对应的名称。

### 4. 常用参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--dataset` | str | `mt_eval` | 指定评测数据集名称 (如 `mathchat`, `longmemeval`) |
| `--raw_data_dir` | str | `./raw_data/MT-Eval` | 原始数据文件路径 |
| `--model_name` | str | `deepseek-ai/DeepSeek-V3.2` | 待评测模型名称 |
| `--model_type` | str | `openai` | 模型后端类型，支持 OpenAI SDK 兼容接口 |
| `--judge_model_name` | str | `gpt-4.1-2025-04-14` | 用于 LLM-as-a-Judge 的裁判模型名称 |
| `--do_generation` | flag | `False` | 是否执行模型回答生成阶段 |
| `--do_evaluation` | flag | `False` | 是否执行指标打分阶段 |
| `--output_dir` | str | `./output` | 结果输出目录 |
| `--parallel` | int | `4` | 并行线程数 |
| `--base_url` | str | `None` | 自定义 API Base URL（用于本地 vLLM 或第三方 API） |
| `--temperature` | float | `0.7` | 生成温度 |
| `--max_tokens` | int | `1024` | 最大生成 token 数 |

## 📂 项目结构

```text
src/
├── dataset/          # 数据集加载与预处理 (支持 mt_eval, mathchat 等)
├── metric/           # 评测指标实现 (Code Match, LLM Judge 等)
├── model/            # 模型接口封装 (OpenAI 等)
├── eval_config.py    # EvalPipelineConfig 配置 dataclass
├── eval_phases.py    # 四阶段逻辑 (Data / Generation / Evaluation / Aggregation)
├── eval_pipeline.py  # EvalPipeline 主控类 & 工厂函数
└── eval_cli.py       # CLI 命令行入口
script/
├── vllm_server.sh    # vLLM 本地模型部署示例
└── test_vllm_client.sh  # vLLM 评测调用示例
```

## 🤝 Get Involved

欢迎对对话评测感兴趣的研究者和开发者参与贡献！如有任何问题、建议或合作意向，欢迎通过以下方式联系我们：

- 📧 Email: qijia0217@gmail.com, aibench.service@gmail.com
- 🐛 Issue: [GitHub Issues](https://github.com/JiaQiSJTU/UniConv-EvalKit/issues)
- 🔀 Pull Request: [GitHub PRs](https://github.com/JiaQiSJTU/UniConv-EvalKit/pulls)

更多评测相关资源，欢迎关注 [https://aiben.ch/home](https://aiben.ch/home)


## 🖊️ Citation

如果您在研究中使用了 UniConv-EvalKit，请引用以下 BibTeX：

```bibtex
@misc{UniConv-EvalKit2026,
  title={UniConv-EvalKit: A Unified Evaluation Toolkit for Comprehensive Conversational Abilities},
  author={xxx},
  year={2026},
  howpublished={\url{https://github.com/JiaQiSJTU/UniConv-EvalKit}}
}
```

<!-- 我们也在多轮对话评测方向开展了系列研究工作，以下评测基准也将陆续集成到本工具中：

- **EvolIF** — 面向多轮指令遵循能力的动态评测基准 [[arxiv](https://arxiv.org/abs/2511.03508v2)] [[code](https://github.com/JiaQiSJTU/EvolIF)]
- **EvolMem** — 面向多轮对话多方面记忆能力的评测基准 [[arxiv](https://arxiv.org/abs/2601.03543)] [[code](https://github.com/shenye7436/EvolMem)] -->




