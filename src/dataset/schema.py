from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MetricConfig(BaseModel):
    """Configuration for a single metric: class_name maps to registry name, args are passed to the metric."""

    class_name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class TurnEvalConfig(BaseModel):
    """Evaluation configuration for each assistant turn."""

    do_eval: bool = False
    metrics: List[MetricConfig] = Field(default_factory=list)

    # Dynamic evaluation config source: used to generate metrics at runtime.
    # If metrics is empty but do_eval=True, the system will try to derive config from dynamic_config_source.
    dynamic_config_source: Dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """
    Unified Turn definition (symmetric structure):
    """

    turn_id: int
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    
    reference: Optional[Union[str, List[str]]] = None
    reference_document: Optional[Union[str, List[int]]] = None
    eval_config: TurnEvalConfig = Field(default_factory=TurnEvalConfig)
    turn_labels: Dict[str, Any] = Field(default_factory=dict)


class DialogEvalConfig(BaseModel):
    """Session-level configuration (extensible)."""

    use_reference_history: bool = False


class Dialog(BaseModel):
    dialog_id: int
    # task_name: str
    dialog_raw_info: Dict[str, Any] = Field(default_factory=dict)
    dialog_labels: Dict[str, Any] = Field(default_factory=dict)
    dialog_eval_config: DialogEvalConfig = Field(default_factory=DialogEvalConfig)
    dialog_turns: List[Turn]


