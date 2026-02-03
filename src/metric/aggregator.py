from typing import Any, Dict, List, Union, Callable
from collections import defaultdict
import statistics
import pandas as pd

def _get_stats_from_series(series: pd.Series) -> Dict[str, float]:
    """Helper to convert a pandas series to the standard stats dictionary."""
    if series.empty:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "min": float(series.min()),
        "max": float(series.max()),
    }

def aggregate_results(
    results: List[Dict[str, Any]], 
    turn_stat: str = "mean",
    dialog_stat: str = "min",
    dataset_level: str = "dialog",
    by_metric: bool = False
) -> Dict[str, Any]:
    """
    Aggregate evaluation results using Pandas for efficient data processing.
    """
    if not results:
        return {"global": {}, "by_metric": {}, "by_label": {}}
    
    # --- Step 1: Load Data into DataFrame ---
    df = pd.DataFrame(results)
    
    # Filter out records without valid scores
    df = df[df['score'].apply(lambda x: isinstance(x, (int, float)))]
    if df.empty:
        return {"global": {}, "by_metric": {}, "by_label": {}}

    # --- Step 2: Hierarchical Aggregation ---
    
    # 2.1 Turn Level Aggregation: did, tid -> aggregated turn score
    # We also keep dialog_labels and turn_labels for later grouping
    turn_df = df.groupby(['dialog_id', 'turn_id']).agg({
        'score': turn_stat,
        'dialog_labels': 'first',
        'turn_labels': 'first'
    }).reset_index()
    
    # 2.2 Dialog Level Aggregation: did -> aggregated dialog score
    dialog_df = turn_df.groupby('dialog_id').agg({
        'score': dialog_stat,
        'dialog_labels': 'first'
    }).reset_index()
    
    # --- Step 3: Global Stats ---
    final_values = dialog_df['score'] if dataset_level == "dialog" else turn_df['score']
    global_stats = _get_stats_from_series(final_values)
    
    # --- Step 4: By Metric Stats ---
    metric_stats = {}
    if by_metric and 'metric_name' in df.columns:
        stats = df.groupby('metric_name')['score'].apply(_get_stats_from_series)
        for val, s in stats.items():
            metric_stats[str(val)] = s

    # --- Step 5: By Label Stats ---
    label_stats = {"dialog": {}, "turn": {}}
    
    # 5.1 Dialog Label Stats
    # Expand nested dialog_labels into columns
    d_labels_df = pd.json_normalize(dialog_df['dialog_labels'])
    if not d_labels_df.empty:
        d_labels_df['score'] = dialog_df['score'].values
        # Iterate through each label column
        for col in d_labels_df.columns:
            if col == 'score': continue
            stats = d_labels_df.groupby(col)['score'].apply(_get_stats_from_series)
            for val, s in stats.items():
                label_stats["dialog"][f"{col}:{val}"] = s
        
        # Specific hierarchy for task_type/subtype
        if 'task_type' in d_labels_df.columns and 'task_subtype' in d_labels_df.columns:
            d_labels_df['task_hierarchy'] = d_labels_df['task_type'].astype(str) + " > " + d_labels_df['task_subtype'].astype(str)
            stats = d_labels_df.groupby('task_hierarchy')['score'].apply(_get_stats_from_series)
            for val, s in stats.items():
                label_stats["dialog"][f"task_hierarchy:{val}"] = s

    # 5.2 Turn Label Stats
    t_labels_df = pd.json_normalize(turn_df['turn_labels'])
    if not t_labels_df.empty:
        t_labels_df['score'] = turn_df['score'].values
        for col in t_labels_df.columns:
            if col == 'score': continue
            stats = t_labels_df.groupby(col)['score'].apply(_get_stats_from_series)
            for val, s in stats.items():
                label_stats["turn"][f"{col}:{val}"] = s

    return {
        "config": {
            "turn_stat": turn_stat,
            "dialog_stat": dialog_stat,
            "dataset_level": dataset_level
        },
        "global": global_stats,
        "by_metric": metric_stats,
        "by_label": label_stats
    }
