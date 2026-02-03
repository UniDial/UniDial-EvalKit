import importlib
import importlib.machinery
import types
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset.mt_eval import MTEvalDataset



def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


    
if __name__ == "__main__":
    dataset = MTEvalDataset(dataset_name="mt_eval")
    dataset.preprocess(raw_path="./raw_data/MT-Eval", processed_root="./tmp/processed", force=True)
    
    for dialog in dataset.load_eval_dialogs(data_root="./tmp/processed"):
        print(dialog)
        break

