"""策略注册表"""

from typing import Dict, Type
from .hunter import HunterStrategy
from .steady import SteadyStrategy
from .base import BaseStrategy

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "hunter": HunterStrategy,
    "steady": SteadyStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """根据名称获取策略实例"""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return cls(**kwargs)
