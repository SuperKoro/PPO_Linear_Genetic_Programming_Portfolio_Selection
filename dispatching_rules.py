# dispatching_rules.py
from typing import Any, Dict, List
from dispatching_registry import register_dr


@register_dr("EDD")
def dr_edd(env,
           finished_events: List[Dict[str, Any]],
           unfinished_jobs: Dict[Any, Any],
           time_budget_s: float = 0.0) -> List[Dict[str, Any]]:
    """
    Dispatching rule EDD: sắp xếp unfinished_jobs theo Earliest Due Date.
    """
    from main import reschedule_unfinished_jobs_edd

    return reschedule_unfinished_jobs_edd(
        unfinished_jobs,
        env.current_time,
        finished_events,
        env.machine_pool
    )


@register_dr("SPT")
def dr_spt(env,
           finished_events: List[Dict[str, Any]],
           unfinished_jobs: Dict[Any, Any],
           time_budget_s: float = 0.0) -> List[Dict[str, Any]]:
    """
    Dispatching rule SPT: sắp xếp theo Shortest Processing Time.
    """
    from main import reschedule_unfinished_jobs_spt

    return reschedule_unfinished_jobs_spt(
        unfinished_jobs,
        env.current_time,
        finished_events,
        env.machine_pool
    )


@register_dr("LPT")
def dr_lpt(env,
           finished_events: List[Dict[str, Any]],
           unfinished_jobs: Dict[Any, Any],
           time_budget_s: float = 0.0) -> List[Dict[str, Any]]:
    """
    Dispatching rule LPT: sắp xếp theo Longest Processing Time.
    """
    from main import reschedule_unfinished_jobs_lpt

    return reschedule_unfinished_jobs_lpt(
        unfinished_jobs,
        env.current_time,
        finished_events,
        env.machine_pool
    )


@register_dr("FCFS")
def dr_fcfs(env,
            finished_events: List[Dict[str, Any]],
            unfinished_jobs: Dict[Any, Any],
            time_budget_s: float = 0.0) -> List[Dict[str, Any]]:
    """
    Dispatching rule FCFS: First Come First Served.
    """
    from main import reschedule_unfinished_jobs_fcfs

    return reschedule_unfinished_jobs_fcfs(
        unfinished_jobs,
        env.current_time,
        finished_events,
        env.machine_pool
    )


@register_dr("FIFO")
def dr_fifo(env,
            finished_events: List[Dict[str, Any]],
            unfinished_jobs: Dict[Any, Any],
            time_budget_s: float = 0.0) -> List[Dict[str, Any]]:
    """
    Dispatching rule FIFO: alias cho FCFS (First In First Out).
    """
    from main import reschedule_unfinished_jobs_fcfs

    return reschedule_unfinished_jobs_fcfs(
        unfinished_jobs,
        env.current_time,
        finished_events,
        env.machine_pool
    )


@register_dr("CR")
def dr_cr(env,
          finished_events: List[Dict[str, Any]],
          unfinished_jobs: Dict[Any, Any],
          time_budget_s: float = 0.0) -> List[Dict[str, Any]]:
    """
    Dispatching rule CR: Critical Ratio.
    """
    from main import reschedule_unfinished_jobs_cr

    return reschedule_unfinished_jobs_cr(
        unfinished_jobs,
        env.current_time,
        finished_events,
        env.machine_pool
    )
