<h1 align="center">
  <img src="assets/logo.jpg" alt="UDE Logo" width="100" height="100" align="absmiddle">
  UniDial-EvalKit
</h1>


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JiaQiSJTU/UniDial-EvalKit.git
   cd UniDial-EvalKit
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n uniconv python=3.10
   conda activate uniconv
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Agent Evaluation: Setup and Usage

This evaluation framework has built-in support for several multi-turn dialogue agents: **HippoRAG**, **MemoryOS**, and **AMem**.

**Note:** These agents are separate codebases. To keep the framework decoupled, you need to place each agent’s core source code in the specified location under `src/model/`. The framework’s adapters will then import and use them.

---

### 1. HippoRAG Agent
HippoRAG is a graph-optimized retrieval-augmented generation agent.

#### 1.1 Download and Directory Setup
Download the HippoRAG source from its official GitHub repository and extract the `hipporag` directory that provides the core functionality.

1. Clone the official HippoRAG repository in a temporary directory:
   ```bash
   git clone https://github.com/OSU-NLP-Group/HippoRAG.git
   ```
2. Copy the **entire** `src/hipporag` folder from the cloned repository into this project’s **`src/model/`** directory. The layout should look like:
   ```text
   UniDial-EvalKit/
   ├── src/
   │   ├── model/
   │   │   ├── hipporag_agent.py   # Our Agent adapter
   │   │   └── hipporag/           # <--- Place here (copied from upstream repo)
   │   │       ├── __init__.py
   │   │       ├── HippoRAG.py
   │   │       └── ...
   ```

#### 1.2 Environment and Dependencies
From the cloned HippoRAG repository root, or using its `requirements.txt`, install its dependencies:
```bash
conda activate uniconv
# If you cloned the repo elsewhere, install its dependencies:
pip install -r /path/to/HippoRAG/requirements.txt
```

#### 1.3 Command-Line Evaluation
After setup, set `--model_type` to `hipporag`. The underlying model is specified with `--model_name`.
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

### 2. MemoryOS Agent
MemoryOS is an agent framework with short-, mid-, and long-term hierarchical memory.

#### 2.1 Download and Directory Setup
1. Clone the official MemoryOS repository:
   ```bash
   git clone https://github.com/BAI-LAB/MemoryOS.git
   ```
2. Take the contents of the `memoryos-pypi` folder (e.g. `memoryos.py`, `retriever.py`, `utils.py`), **rename the folder to `memoryos`**, and place it under this project’s **`src/model/`** directory:
   ```text
   UniDial-EvalKit/
   ├── src/
   │   ├── model/
   │   │   ├── memoryos_agent.py   # Framework adapter
   │   │   └── memoryos/           # <--- From upstream memoryos-pypi, copied and renamed
   │   │       ├── __init__.py
   │   │       ├── memoryos.py
   │   │       ├── utils.py        # (Required; missing utils.py will cause import errors)
   │   │       └── ...
   ```

#### 2.2 Environment and Dependencies
From the project root, install MemoryOS’s dependencies (under `src/model/memoryos`):
```bash
conda activate uniconv
pip install -r src/model/memoryos/requirements.txt
```

#### 2.3 Command-Line Evaluation
Set `--model_name` to your LLM and `--model_type` to `memoryos`.
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

### 3. AMem Agent
AgenticMemory (AMem) is a lightweight agentic memory management system.

#### 3.1 Download and Directory Setup
Obtain the A-mem project source and place it in the correct location.

1. Get the A-mem core code from its GitHub repo (or source archive):
   ```bash
   # Example: clone via Git
   git clone https://github.com/WujiangXu/A-mem.git
   ```
2. Take the core memory package (the folder that contains `memory_layer.py`, e.g. from `A-mem-main` or `A-mem`), **rename it to `AgenticMemory`**, and place it under this project’s **`src/model/`** directory:
   ```text
   UniDial-EvalKit/
   ├── src/
   │   ├── model/
   │   │   ├── amem_agent.py       # Framework adapter
   │   │   └── AgenticMemory/      # <--- Place here; folder name must be AgenticMemory
   │   │       ├── memory_layer.py # Must contain this core module
   │   │       └── ...
   ```

#### 3.2 Environment and Dependencies
AMem depends on specific third-party libraries. From the project root, install dependencies for `src/model/AgenticMemory`:
```bash
conda activate uniconv
pip install -r src/model/AgenticMemory/requirements.txt
```

#### 3.3 Command-Line Evaluation
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
