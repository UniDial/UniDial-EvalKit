# MTEvalKit

**MTEvalKit** 是一个专注于 **大模型多轮交互评测** 的统一框架。我们致力于构建覆盖 **记忆、理解、安全、数学、代码** 等多维度的长期交互综合能力评价体系。通过集成多个主流 Benchmark 进行综合对比，MTEvalKit 为大语言模型及 Agent 系统的多轮人机交互技术演进提供全方位、标准化的能力画像。

作为一款模块化、可扩展的评测工具，MTEvalKit 将数据加载、模型生成与指标评测解耦，支持断点续传与并行加速，能够高效地对各类开源及闭源模型进行全方位能力评估。

## ✨ 项目亮点

- **模块化架构**：核心组件（`Dataset`, `Model`, `Metric`）完全解耦，易于扩展新的数据集或评测指标。
- **高效并行**：支持多线程并行生成与评测，大幅缩短大规模评估时间。
- **断点续传**：内置结果缓存机制，自动跳过已完成的对话或评测条目，避免意外中断导致的重复计算。
- **丰富的评测基准**：内置多种前沿评测数据集支持，包括但不限于：
  - **通用能力**：`mt_eval`, `mt_bench_101`, `multi_challenge`
  - **记忆**：`longmemeval`, `locomo`
  - **特定任务**：`mathchat` (数学), `memorycode`(代码), `multi_if`（指令遵循）, `personamem`(个性化)
- **灵活的聚合统计**：支持从 Turn (轮次) 到 Dialog (对话) 再到 Dataset (数据集) 的多级分数聚合策略（Mean/Min/Max）。

## 🛠️ 安装方法

1. 克隆本项目代码：
   ```bash
   git clone https://github.com/your_username/MTEvalKit.git
   cd MTEvalKit/code
   ```

2. 安装依赖环境：
   ```bash
   pip install -r requirements.txt
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

