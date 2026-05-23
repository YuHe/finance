"""策略注册表"""

from typing import Dict, Type
from .adaptive_premium import AdaptivePremiumStrategy
from .momentum_quality import MomentumQualityStrategy
from .base import BaseStrategy

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "adaptive_premium": AdaptivePremiumStrategy,
    "momentum_quality": MomentumQualityStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """根据名称获取策略实例"""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return cls(**kwargs)
