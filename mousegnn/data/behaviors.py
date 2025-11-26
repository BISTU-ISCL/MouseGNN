"""Behavior definitions and helper utilities for mouse action recognition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

BEHAVIORS: List[str] = [
    "walk",       # 行走
    "circle",     # 转圈
    "sniff",      # 嗅探
    "rear",       # 直立
    "curl",       # 蜷缩
]


def behavior_to_index() -> Dict[str, int]:
    """Return a mapping from behavior string to index.

    The ordering is defined by :data:`BEHAVIORS` to guarantee reproducibility
    between annotation, training, and evaluation.
    """

    return {name: idx for idx, name in enumerate(BEHAVIORS)}


@dataclass
class BehaviorNames:
    walk: str = "walk"
    circle: str = "circle"
    sniff: str = "sniff"
    rear: str = "rear"
    curl: str = "curl"

    def as_list(self) -> List[str]:
        return [self.walk, self.circle, self.sniff, self.rear, self.curl]


DEFAULT_BEHAVIORS = BehaviorNames()
