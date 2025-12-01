# typed_action_adapter.py
from __future__ import annotations
from typing import Any, Dict, List

from lgp_actions import ActionIndividual, Gene
from dispatching_registry import has_dr, get_dr
from mh_registry import has_mh, get_mh


def run_action_individual(env,
                          individual: ActionIndividual,
                          finished_events: List[Dict[str, Any]],
                          unfinished_jobs: Dict[Any, Any],
                          total_budget_s: float) -> List[Dict[str, Any]]:
    """
    Chạy 1 ActionIndividual trên env:
    - Stage 0: áp dụng DR để sắp xếp lại unfinished_jobs.
    - Stage 1..n: áp dụng các MH tuần tự theo vector weight.
    """
    current_time = env.current_time
    all_jobs_info = env.all_jobs_info

    from main import split_schedule_list  # tránh circular import

    # --------- Stage 0: Dispatching Rule ----------
    dr_gene: Gene = individual.dr_gene
    dr_name = dr_gene.name.upper()

    if has_dr(dr_name):
        dr_fn = get_dr(dr_name)
    else:
        if has_dr("EDD"):
            dr_fn = get_dr("EDD")
        else:
            raise RuntimeError("Không tìm thấy dispatching rule hợp lệ (EDD).")

    cand_unfinished = dr_fn(
        env,
        finished_events=finished_events,
        unfinished_jobs=unfinished_jobs,
        time_budget_s=0.0
    )
    virtual_events = finished_events + cand_unfinished
    finished, unfinished = split_schedule_list(virtual_events, current_time, all_jobs_info)

    # --------- Stage 1..n: Metaheuristic ----------
    mh_genes: List[Gene] = individual.mh_genes

    raw_ws = [max(0.0, g.w_raw) for g in mh_genes]
    if sum(raw_ws) <= 0.0:
        raw_ws = [1.0] * len(mh_genes)
    total_w = float(sum(raw_ws))

    last_unfinished = cand_unfinished

    for gene, w_raw in zip(mh_genes, raw_ws):
        mh_name = gene.name.upper()
        if not has_mh(mh_name):
            continue

        mh_fn = get_mh(mh_name)
        stage_budget = (w_raw / total_w) * float(total_budget_s)

        if stage_budget <= 1e-9:
            continue

        cand_unfinished = mh_fn(
            env,
            finished_events=finished,
            unfinished_jobs=unfinished,
            time_budget_s=stage_budget
        )
        last_unfinished = cand_unfinished

        virtual_events = finished + cand_unfinished
        finished, unfinished = split_schedule_list(virtual_events, current_time, all_jobs_info)

    return last_unfinished
