# lgp_actions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import random


@dataclass
class Gene:
    """
    Gene cho 1 rule:
    - kind: "DR" hoặc "MH"
    - name: "EDD", "SA", "GA", "PSO", ...
    - w_raw: weight thô (dùng cho MH)
    """
    kind: str
    name: str
    w_raw: float = 1.0


@dataclass
class ActionIndividual:
    """
    1 cá thể hành động (portfolio):
    - genes[0]: DR gene
    - genes[1..]: MH genes
    """
    genes: List[Gene]

    def __post_init__(self):
        if len(self.genes) < 2:
            raise ValueError("ActionIndividual phải có ít nhất 1 DR + 1 MH.")
        if self.genes[0].kind.upper() != "DR":
            raise ValueError("Gene 0 phải có kind='DR'.")

    @property
    def dr_gene(self) -> Gene:
        return self.genes[0]

    @property
    def mh_genes(self) -> List[Gene]:
        return self.genes[1:]


def individual_normalized_weights(individual: ActionIndividual) -> List[float]:
    """
    Trả về vector weight đã chuẩn hoá (tổng = 1) cho các MH gene của cá thể.
    KHÔNG đổi logic hệ thống, chỉ dùng để đọc/log cho dễ.
    """
    mh_genes = individual.mh_genes
    raw_ws = [max(0.0, g.w_raw) for g in mh_genes]
    total = sum(raw_ws)
    if total <= 0.0:
        # nếu cá thể có toàn weight <=0, ta chia đều để đọc cho dễ
        return [1.0 / len(raw_ws)] * len(raw_ws)
    return [w / total for w in raw_ws]


def describe_individual(individual: ActionIndividual) -> str:
    """
    Trả về string mô tả portfolio dạng:
    DR=EDD | SA(raw=3.40,norm=0.50) ; GA(raw=1.70,norm=0.25) ; PSO(raw=1.70,norm=0.25)
    """
    dr = individual.dr_gene
    mh_genes = individual.mh_genes
    norm_ws = individual_normalized_weights(individual)

    parts = []
    for g, w_norm in zip(mh_genes, norm_ws):
        # w_raw có thể >1, không sao; w_norm luôn [0,1]
        parts.append(f"{g.name}(raw={g.w_raw:.2f}, norm={w_norm:.2f})")

    return f"DR={dr.name} | " + " ; ".join(parts)


class ActionLGP:
    """
    LGP đơn giản để KHỞI TẠO pool các ActionIndividual ngẫu nhiên.
    Việc tiến hoá pool được làm ở coevolution_trainer.
    """
    def __init__(self,
                 dr_list: Sequence[str],
                 mh_list: Sequence[str],
                 pool_size: int = 64,
                 n_mh_genes: int = 3,
                 seed: int | None = 0):
        if not dr_list:
            raise ValueError("Cần ít nhất 1 dispatching rule trong dr_list.")
        if not mh_list:
            raise ValueError("Cần ít nhất 1 metaheuristic trong mh_list.")

        self.rng = random.Random(seed)
        self.dr_list = [d.upper() for d in dr_list]
        self.mh_list = [m.upper() for m in mh_list]
        self.pool_size = int(pool_size)
        self.n_mh_genes = int(n_mh_genes)

        self.pool: List[ActionIndividual] = [
            self._random_individual()
            for _ in range(self.pool_size)
        ]

    def _random_individual(self) -> ActionIndividual:
        g0 = Gene(
            kind="DR",
            name=self.rng.choice(self.dr_list),
            w_raw=1.0
        )
        genes: List[Gene] = [g0]

        for _ in range(self.n_mh_genes):
            genes.append(
                Gene(
                    kind="MH",
                    name=self.rng.choice(self.mh_list),
                    w_raw=self.rng.uniform(0.1, 1.5)
                )
            )
        return ActionIndividual(genes=genes)
