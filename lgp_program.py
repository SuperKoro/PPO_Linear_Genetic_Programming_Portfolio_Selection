from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ga_portfolio import ActionIndividual, Gene
from config import LGPConfig
from lgp_instructions import (
    Instruction,
    SetPortfolioInstruction,
    instruction_from_dict,
)


class PortfolioBuilder:
    """
    Helper để build ActionIndividual từ các lệnh SET_DR / SET_MH*.
    """

    def __init__(
        self,
        default_dr: str = "EDD",
        default_mh: str = "SA",
        n_mh: int = None,
    ) -> None:
        self.default_dr = default_dr
        self.default_mh = default_mh
        self.n_mh = n_mh or LGPConfig.n_mh_genes
        self.dr_name: Optional[str] = None
        # component -> (mh_name, weight)
        self._mh_slots: Dict[str, tuple[str, float]] = {}

    def set_mh(self, component: str, mh_name: str, weight: float) -> None:
        w = float(weight)
        if w < 0.0:
            w = 0.0
        self._mh_slots[component] = (mh_name, w)

    def set_dr(self, dr_name: str) -> None:
        self.dr_name = dr_name

    def build(self) -> ActionIndividual:
        # DR gene
        dr = self.dr_name or self.default_dr
        genes: List[Gene] = [Gene(kind="DR", name=dr, w_raw=1.0)]

        # MH genes theo thứ tự MH1..MHn
        for i in range(self.n_mh):
            comp = f"MH{i+1}"
            if comp in self._mh_slots:
                mh_name, w = self._mh_slots[comp]
            else:
                mh_name, w = self.default_mh, 0.0
            genes.append(Gene(kind="MH", name=mh_name, w_raw=w))

        return ActionIndividual(genes=genes)


@dataclass
class LGPProgram:
    """
    Linear GP program = danh sách instruction chạy trên dãy registers.
    """
    instructions: List[Instruction]
    num_registers: int = 20

    def execute(self, inputs: Dict[str, float]) -> ActionIndividual:
        """
        Chạy program với inputs và build portfolio (ActionIndividual).
        """
        registers: List[float] = [0.0 for _ in range(self.num_registers)]

        ctx: Dict[str, Any] = {
            "inputs": inputs,
            "available_dr": LGPConfig.available_dr,
            "available_mh": LGPConfig.available_mh,
        }
        builder = PortfolioBuilder(
            default_dr=LGPConfig.available_dr[0],
            default_mh=LGPConfig.available_mh[0],
            n_mh=LGPConfig.n_mh_genes,
        )
        ctx["builder"] = builder

        ip = 0
        n = len(self.instructions)
        while ip < n:
            instr = self.instructions[ip]
            skip = instr.execute(registers, ctx)
            if skip is None:
                skip = 0
            ip += 1 + max(0, int(skip))

        return builder.build()

    def clone(self) -> "LGPProgram":
        return LGPProgram(
            instructions=[instr.clone() for instr in self.instructions],
            num_registers=self.num_registers,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_registers": self.num_registers,
            "instructions": [instr.to_dict() for instr in self.instructions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LGPProgram":
        inst_dicts = data.get("instructions", [])
        instructions = [instruction_from_dict(d) for d in inst_dicts]
        num_registers = int(data.get("num_registers", 20))
        return cls(instructions=instructions, num_registers=num_registers)
