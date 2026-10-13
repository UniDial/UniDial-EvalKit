# UniDial-EvalKit: A Unified Toolkit for Evaluating Multi-Faceted Conversational Abilities

**UniDial-EvalKit** (UDE) is a unified framework focusing on the **evaluation of multi-turn interactions in Large Language Models (LLMs)**. We are committed to building a comprehensive evaluation system for long-term interactions covering dimensions such as **memory, understanding, safety, mathematics, and code**. By integrating multiple mainstream benchmarks for comprehensive comparison, UniDial-EvalKit provides a full-scale, standardized capability profile for the evolution of multi-turn human-AI interaction technologies in LLMs and Agent systems.

As a modular and extensible evaluation tool, UniDial-EvalKit decouples data loading, model generation, and metric evaluation. It supports breakpoint resume and parallel acceleration, enabling efficient all-round assessment of various open-source and closed-source models.

## ✨ Project Highlights

- **Modular Architecture**: Core components (`Dataset`, `Model`, `Metric`) are completely decoupled, making it easy to extend new datasets, evaluation targets, or metrics.
- **Efficient Parallelism**: Supports multi-threaded parallel generation and evaluation, significantly reducing the time required for large-scale evaluation.
- **Breakpoint Resume**: Built-in result caching mechanism automatically skips completed dialogues or evaluation items, avoiding redundant inference and computation caused by unexpected interruptions.
- **Rich Benchmarks**: Built-in support for various cutting-edge evaluation datasets, including but not limited to:
  - **General Capabilities**: `mt_eval`, `mt_bench_101`, `multi_challenge`
  - **Memory**: `longmemeval`, `locomo`
  - **Safety**: `safedialbench`
  - **Task-Specific**: `mathchat` (Math), `memorycode` (Code), `multi_if` (Instruction Following), `personamem` (Personalization)
- **Flexible Aggregation**: Supports multi-level score aggregation strategies (Mean/Min/Max) from **Turn** to **Dialog** to **Dataset**.


## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone xxx
   cd UniDial-EvalKit
   ```

2. Create and activate a conda environment:

    ```bash
    conda create -n unidial python=3.10
    conda activate unidial
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## 📦 Data Preparation

Before running the evaluation, you need to place the raw data files for the corresponding dataset in the `raw_data/` directory. The required files for each dataset are as follows:


| Benchmark | `--dataset` | Dimension | Data Source | Reference |
| :--- | :--- | :--- | :--- | :--- |
| MT-Eval | `mt_eval` | General | [🤗 Hugging Face](https://huggingface.co/datasets/wckwan/MT-Eval)(`*.jsonl`) | [2024EMNLP](https://aclanthology.org/2024.emnlp-main.1124.pdf) |
| MT-Bench-101 | `mt_bench_101` | General | [:octocat: Github](https://github.com/mtbench101/mt-bench-101)(`*.jsonl`) | [2024ACL](https://aclanthology.org/2024.acl-long.401/) |
| Multi-Challenge | `multi_challenge` | General | [:octocat: Github](https://github.com/ekwinox117/multi-challenge)(`*.jsonl`) | [2025ACLfindings](https://aclanthology.org/2025.findings-acl.958/) |
| LongMemEval | `longmemeval` | Memory | [🤗 Hugging Face](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)(`*.json`) | [2025ICLR Poster](https://openreview.net/pdf?id=pZiyCaVuti) |
| LoCoMo | `locomo` | Memory | [:octocat: Github](https://github.com/snap-research/locomo/tree/main/data)(`*.json`) | [2024ACL](https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2024.acl-long.747.pdf) |
| MathChat | `mathchat` | Math | [:octocat: Github](https://github.com/Zhenwen-NLP/MathChat)(`*.jsonl`) | [Arxiv](https://arxiv.org/pdf/2405.19444) |
| MemoryCode | `memorycode` | Code | [:octocat: Github](https://github.com/Cohere-Labs-Community/MemoryCode)(`dialogue_*.json`) | [2025ACL](https://arxiv.org/abs/2502.13791) |
| Multi-IF | `multi_if` | Instruction Following | [🤗 Hugging Face](https://huggingface.co/datasets/facebook/Multi-IF)(`*.csv`) | [2025ACL](https://aclanthology.org/2025.acl-long.1172.pdf) |
| PersonaMem | `personamem` | Personalization | [🤗 Hugging Face](https://huggingface.co/datasets/bowen-upenn/PersonaMem)(`questions_*.csv`, `shared_contexts_*.jsonl`) | [2025COLM](https://arxiv.org/abs/2504.14225) |
| SafeDialBench | `safedialbench` | Safety | [🤗 Hugging Face](https://huggingface.co/datasets/HongyeCao/SafeDialBench)(`datasets_*.jsonl`& prompt file `*.jsonl`) | [2026ICLR](https://arxiv.org/abs/2502.11090) |

After placing the corresponding files into `raw_data/<directory_name>/`, specify the path using the `--raw_data_dir` parameter, for example:

```bash
--raw_data_dir "./raw_data/MT-Eval"
```

## 🚀 Usage

The evaluation workflow is primarily divided into four stages: **Data Loading**, **Model Generation**, **Metric Evaluation**, and **Result Aggregation**. It supports both **CLI** and **Python API**.

### 1. CLI Command Line
The entry script is `src/eval_cli.py`, which is flexibly controlled via command-line arguments.

**Full Workflow (Generation + Evaluation):**

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

**Generation Only:**

```bash
PYTHONPATH=. python src/eval_cli.py \
    --dataset mathchat \
    --raw_data_dir ./raw_data/MathChat \
    --model_name deepseek-chat \
    --base_url xxx \
    --api_key "your-api-key" \
    --do_generation
```

**Evaluation Only (Requires existing generation results):**

```bash
PYTHONPATH=. python src/eval_cli.py \
    --dataset mathchat \
    --model_name deepseek-chat \
    --judge_model_name gpt-4 \
    --base_url xxx \
    --api_key "your-api-key" \
    --do_evaluation
```


### 2. Python API

Using `EvalPipeline` and `EvalPipelineConfig`, you can call the framework within scripts or Notebooks, allowing for fine-grained control over each stage.


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

# One-click execution
pipeline = EvalPipeline(cfg)
pipeline.run()

# Or call by stage (supports custom intermediate processing)
# dialogs   = pipeline.prepare_data()
# generated = pipeline.run_generation(dialogs)
# results   = pipeline.run_evaluation(generated)
# summary   = pipeline.run_aggregation(results)
```

### 3. Deploying Local Models with vLLM

For open-source models, you can deploy an OpenAI-compatible inference service via [vLLM](https://docs.vllm.ai/) and then connect it directly to this framework for evaluation.

**Step 1: Start vLLM service** (see `script/vllm_server.sh`)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model ./Models/Qwen3-8B \
    --served-model-name "Qwen3-8B" \
    --port 8000
```

The service listens on `http://localhost:8000` by default.

**Step 2: Run evaluation**, pointing to the local vLLM service via `--base_url` (see `script/test_vllm_client.sh`)

```bash
PYTHONPATH=. python src/eval_cli.py \
    --dataset locomo \
    --raw_data_dir ./raw_data/LoCoMo \
    --model_name Qwen3-8B \
    --base_url http://localhost:8000/v1/ \
    --do_generation \
    --api_key 'x'
```

> 💡 Since vLLM provides an OpenAI-compatible interface, you can keep the default `--model_type` as `openai.` Just point `--base_url` to your local address and set `--model_name` to the corresponding `--served-model-name`, and you can pass any non-empty string as the api_key.

### 4. Supported Agents

Currently, the framework integrates A-Mem, HippoRAG, and MemoryOS agents. For detailed setup and usage instructions, please refer to [`src/model/README_agent.md`](src/model/README_agent.md).


### 5. Common Parameter Descriptions


| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | str | `mt_eval` | Name of the evaluation dataset (e.g., `mathchat`, `longmemeval`) |
| `--raw_data_dir` | str | `./raw_data/MT-Eval` | Path to the raw data files |
| `--model_name` | str | `deepseek-ai/DeepSeek-V3.2` | Name of the model to be evaluated |
| `--model_type` | str | `openai` | Model backend type, supports OpenAI SDK compatible interfaces |
| `--judge_model_name` | str | `gpt-4.1-2025-04-14` | Name of the judge model for LLM-as-a-Judge |
| `--do_generation` | flag | `False` | Whether to execute the response generation phase |
| `--do_evaluation` | flag | `False` | Whether to execute the metric scoring phase |
| `--output_dir` | str | `./output` | Directory for output results |
| `--parallel` | int | `4` | Number of parallel threads |
| `--base_url` | str | `None` | Custom API Base URL (for local vLLM or third-party APIs) |
| `--temperature` | float | `0.7` | Generation temperature |
| `--max_tokens` | int | `1024` | Maximum number of tokens to generate |


## 📂 Project Structure

```text
src/
├── dataset/          # Dataset loading and preprocessing (supports mt_eval, mathchat, etc.)
├── metric/           # Implementation of evaluation metrics (Code Match, LLM Judge, etc.)
├── model/            # Model interface wrappers (OpenAI, etc.)
├── eval_config.py    # EvalPipelineConfig configuration dataclass
├── eval_phases.py    # Logic for the four phases (Data / Generation / Evaluation / Aggregation)
├── eval_pipeline.py  # EvalPipeline main controller class & factory functions
└── eval_cli.py       # CLI command-line entry point
script/
├── vllm_server.sh    # Example for local vLLM model deployment
└── test_vllm_client.sh  # Example for vLLM evaluation call
```