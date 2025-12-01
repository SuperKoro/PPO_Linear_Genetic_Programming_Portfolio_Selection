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
    from main import reschedule_unfinished_jobs_edd  # dùng hàm có sẵn trong main

    return reschedule_unfinished_jobs_edd(
        unfinished_jobs,
        env.current_time,
        finished_events,
        env.machine_pool
    )
