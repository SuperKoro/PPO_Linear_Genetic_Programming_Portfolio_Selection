# metaheuristics_impl.py
from typing import Any, Dict, List
from mh_registry import register_mh


@register_mh("SA")
def mh_sa(env,
          finished_events: List[Dict[str, Any]],
          unfinished_jobs: Dict[Any, Any],
          time_budget_s: float):
    """
    Metaheuristic SA: gọi reschedule_unfinished_jobs_sa trong main.py.
    Bạn có thể tinh chỉnh mapping time_budget_s -> iterations.
    """
    from main import reschedule_unfinished_jobs_sa
    current_time = env.current_time
    machine_pool = env.machine_pool

    iterations = max(10, int(time_budget_s * 10.0))
    return reschedule_unfinished_jobs_sa(
        unfinished_jobs,
        current_time,
        finished_events,
        machine_pool,
        iterations=iterations
    )


@register_mh("GA")
def mh_ga(env,
          finished_events: List[Dict[str, Any]],
          unfinished_jobs: Dict[Any, Any],
          time_budget_s: float):
    from main import reschedule_unfinished_jobs_ga
    current_time = env.current_time
    machine_pool = env.machine_pool

    num_candidates = max(5, int(time_budget_s * 5.0))
    generations = max(3, int(time_budget_s * 3.0))
    return reschedule_unfinished_jobs_ga(
        unfinished_jobs,
        current_time,
        finished_events,
        machine_pool,
        num_candidates=num_candidates,
        generations=generations
    )


@register_mh("PSO")
def mh_pso(env,
           finished_events: List[Dict[str, Any]],
           unfinished_jobs: Dict[Any, Any],
           time_budget_s: float):
    from main import reschedule_unfinished_jobs_pso
    current_time = env.current_time
    machine_pool = env.machine_pool

    num_particles = max(5, int(time_budget_s * 5.0))
    iterations = max(5, int(time_budget_s * 5.0))
    return reschedule_unfinished_jobs_pso(
        unfinished_jobs,
        current_time,
        finished_events,
        machine_pool,
        num_particles=num_particles,
        iterations=iterations
    )


@register_mh("EDD")
def mh_edd(env,
           finished_events: List[Dict[str, Any]],
           unfinished_jobs: Dict[Any, Any],
           time_budget_s: float):
    """
    Đăng ký EDD như 1 "MH" để có thể dùng nó trong vector weight.
    Thực chất vẫn là reschedule_unfinished_jobs_edd.
    """
    from main import reschedule_unfinished_jobs_edd
    current_time = env.current_time
    machine_pool = env.machine_pool
    return reschedule_unfinished_jobs_edd(
        unfinished_jobs,
        current_time,
        finished_events,
        machine_pool
    )
