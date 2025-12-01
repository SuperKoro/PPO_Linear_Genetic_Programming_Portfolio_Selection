# mh_registry.py
from typing import Callable, Dict, Any

# name -> callable(env, finished_events, unfinished_jobs, time_budget_s) -> new_unfinished_events
MH_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_mh(name: str):
    """
    Decorator để đăng ký 1 metaheuristic.
    Hàm MH phải có dạng:
        fn(env, finished_events, unfinished_jobs, time_budget_s) -> new_unfinished_events
    """
    name = name.upper()

    def deco(fn: Callable[..., Any]):
        MH_REGISTRY[name] = fn
        return fn

    return deco


def has_mh(name: str) -> bool:
    return name.upper() in MH_REGISTRY


def get_mh(name: str) -> Callable[..., Any]:
    return MH_REGISTRY[name.upper()]
