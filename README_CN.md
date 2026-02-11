# MTEvalKit

**MTEvalKit** 是一个专注于 **大模型多轮交互评测** 的统一框架。我们致力于构建覆盖 **记忆、理解、安全、数学、代码** 等多维度的长期交互综合能力评价体系。通过集成多个主流Benchmark进行综合对比，MTEvalKit为大语言模型及 Agent系统的多轮人机交互技术演进提供全方位、标准化的能力画像。

作为一款模块化、可扩展的评测工具，MTEvalKit 将数据加载、模型生成与指标评测解耦，支持断点续传与并行加速，能够高效地对各类开源及闭源模型进行全方位能力评估。

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

## 🛠️ 安装方法

1. 克隆本项目代码：
   ```bash
   git clone https://github.com/JiaQiSJTU/MTEvalKit.git
   cd MTEvalKit
   ```

2. 安装依赖环境：
   ```bash
   pip install -r requirements.txt
   ```

## 📦 数据准备

在运行评测前，需要将对应数据集的原始数据文件放置在 `raw_data/` 目录下。各数据集所需文件如下：

| Benchmark | `--dataset` | 能力维度 | 数据源 | 参考文献 |
| :--- | :--- | :--- | :--- | :--- |
| MT-Eval | `mt_eval` | 通用能力 | [🤗 MT-Eval](https://huggingface.co/datasets/wckwan/MT-Eval)(`*.jsonl`) | [2024EMNLP](https://aclanthology.org/2024.emnlp-main.1124.pdf) |
| MT-Bench-101 | `mt_bench_101` | 通用能力 | [MT-Bench-101](https://github.com/mtbench101/mt-bench-101)(`*.jsonl`) | [2024ACL](https://aclanthology.org/2024.acl-long.401/) |
| Multi-Challenge | `multi_challenge` | 通用能力 | [Multi-Challenge](https://github.com/ekwinox117/multi-challenge)(`*.jsonl`) | [2025ACLfindings](https://aclanthology.org/2025.findings-acl.958/) |
| LongMemEval | `longmemeval` | 记忆 | [LongMemEval](https://github.com/xiaowu0162/LongMemEval)(`*.json`) | [2025ICLR Poster](https://openreview.net/pdf?id=pZiyCaVuti) |
| LoCoMo | `locomo` | 记忆 | [Locomo](https://snap-research.github.io/locomo)(`*.json`) | [2024ACL](https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2024.acl-long.747.pdf) |
| MathChat | `mathchat` | 数学 | [MathChat](https://github.com/Zhenwen-NLP/MathChat)(`*.jsonl`) | [Arxiv](https://arxiv.org/pdf/2405.19444) |
| MemoryCode | `memorycode` | 代码 | [MemoryCode](https://github.com/Cohere-Labs-Community/MemoryCode)(`dialogue_*.json`) | [ACL2025](https://arxiv.org/abs/2502.13791) |
| Multi-IF | `multi_if` | 指令遵循 | [Multi-IF](https://github.com/facebookresearch/Multi-IF)(`*.csv`) | [2025ACL](https://aclanthology.org/2025.acl-long.1172.pdf) |
| PersonaMem | `personamem` | 个性化 | [PersonaMem](http://github.com/bowen-upenn/PersonaMem)(`questions_*.csv`, `shared_contexts_*.jsonl`) | [2025COLM](https://arxiv.org/abs/2504.14225) |
| SafeDialBench | `safedialbench` | 安全 | [🤗 SafeDialBench](https://huggingface.co/datasets/HongyeCao/SafeDialBench)(`datasets_*.jsonl`及prompt文件`*.jsonl`) | [2026ICLR](https://arxiv.org/abs/2502.11090) |

将对应文件放入 `raw_data/<目录名>/` 后，通过 `--raw_data_dir` 参数指定路径即可，例如：

```bash
--raw_data_dir "./raw_data/MT-Eval"
```

## 🚀 使用方法

评测流程主要分为 **生成 (Generation)** 和 **评测 (Evaluation)** 两个阶段，可以通过命令行参数灵活控制。入口脚本为 `src/eval.py`。

### 1. 快速开始

运行以下命令即可完成从生成到评测的全流程：

```bash
python src/eval.py \
    --dataset mt_eval \
    --model_type openai \
    --model_name gpt-3.5-turbo \
    --judge_model_name gpt-4 \
    --api_key "your-api-key" \
    --do_generation \
    --do_evaluation \
    --parallel 4
```

### 2. 常用参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--dataset` | str | `mt_eval` | 指定评测数据集名称 (如 `mathchat`, `longmemeval`) |
| `--model_name` | str | `deepseek-chat` | 待评测模型名称 |
| `--model_type` | str | `openai` | 模型后端类型，支持 OpenAI SDK 兼容接口 |
| `--judge_model_name` | str | `deepseek-chat` | 用于 LLM-as-a-Judge 的裁判模型名称 |
| `--do_generation` | flag | `False` | 是否执行模型回答生成阶段 |
| `--do_evaluation` | flag | `False` | 是否执行指标打分阶段 |
| `--output_dir` | str | `./output` | 结果输出目录 |
| `--parallel` | int | `4` | 并行线程数 |
| `--base_url` | str | `None` | 自定义 API Base URL (例如用于本地 vLLM 或 DeepSeek API) |

### 3. 分阶段运行

**仅生成回答：**
```bash
python src/eval.py --dataset mathchat --model_name xxx --base_url xxx --api_key xxx --do_generation
```

**仅运行评测（需已有生成结果）：**
```bash
python src/eval.py --dataset mathchat --model_name xxx --base_url xxx --api_key xxx --do_evaluation
```

## 📂 项目结构

```text
src/
├── dataset/      # 数据集加载与预处理 (支持 mt_eval, mathchat 等)
├── metric/       # 评测指标实现 (Code Match, LLM Judge 等)
├── model/        # 模型接口封装 (OpenAI 等)
└── eval.py       # 评测主程序入口
```

## 🖊️ Citation

如果您在研究中使用了 MTEvalKit，请引用以下 BibTeX：

```bibtex
@misc{MTEvalKit2026,
  title={MTEvalKit: Multi-Turn Evaluation Toolkit for Comprehensive Dialogue Abilities},
  author={},
  year={2026},
  howpublished={\url{https://github.com/JiaQiSJTU/MTEvalKit}}
}
```

