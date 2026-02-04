from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MetricConfig(BaseModel):
    """单个指标的配置：class_name 对应 Registry 名称，args 传给指标。"""

    class_name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class TurnEvalConfig(BaseModel):
    """每一轮 assistant turn 的评测配置。"""

    do_eval: bool = False
    metrics: List[MetricConfig] = Field(default_factory=list)

    # 动态评测配置源数据：用于运行时生成 metrics
    # 如果 metrics 为空但 do_eval=True，则尝试从 dynamic_config_source 中获取配置
    dynamic_config_source: Dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """
    统一的 Turn 定义（对称结构）：
    - JSON 中缺失字段会被 Pydantic 用默认值补齐
    - eval_config 默认永远存在，避免 AttributeError
    """

    turn_id: int
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    
    reference: Optional[Union[str, List[str]]] = None
    reference_document: Optional[Union[str, List[int]]] = None
    eval_config: TurnEvalConfig = Field(default_factory=TurnEvalConfig)
    turn_labels: Dict[str, Any] = Field(default_factory=dict)


class DialogEvalConfig(BaseModel):
    """Session 级配置（可扩展）。"""

    use_reference_history: bool = False


class Dialog(BaseModel):
    dialog_id: int
    # task_name: str
    dialog_raw_info: Dict[str, Any] = Field(default_factory=dict)
    dialog_labels: Dict[str, Any] = Field(default_factory=dict)
    dialog_eval_config: DialogEvalConfig = Field(default_factory=DialogEvalConfig)
    dialog_turns: List[Turn]


