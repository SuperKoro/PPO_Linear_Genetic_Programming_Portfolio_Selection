# dispatching_registry.py
from typing import Callable, Dict, Any

DR_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_dr(name: str):
    """
    Decorator để đăng ký 1 dispatching rule.
    Hàm được đăng ký có dạng:
        fn(env, finished_events, unfinished_jobs, time_budget_s=0.0) -> new_unfinished_events
    """
    name = name.upper()

    def deco(fn: Callable[..., Any]):
        DR_REGISTRY[name] = fn
        return fn

    return deco


def has_dr(name: str) -> bool:
    return name.upper() in DR_REGISTRY


def get_dr(name: str) -> Callable[..., Any]:
    return DR_REGISTRY[name.upper()]
