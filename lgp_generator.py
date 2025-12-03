from __future__ import annotations
from typing import List
import random

from config import LGPConfig
from lgp_instructions import (
    Instruction,
    LoadInputInstruction,
    LoadConstInstruction,
    CopyRegInstruction,
    ArithmeticInstruction,
    ClampMinInstruction,
    ClampMaxInstruction,
    ConditionalSkipInstruction,
    SetPortfolioInstruction,
)
from lgp_program import LGPProgram


# Các key input macro, bạn có thể mở rộng thêm
DEFAULT_INPUT_KEYS = [
    "num_jobs",
    "avg_processing_time",
    "avg_ops_per_job",
]


class LGPGenerator:
    """
    Random program generator cho Linear GP.
    """

    def __init__(
        self,
        max_length: int = 30,
        min_length: int = 8,
        num_registers: int = 20,
        rng: random.Random | None = None,
        input_keys: List[str] | None = None,
    ) -> None:
        self.max_length = max_length
        self.min_length = min_length
        self.num_registers = num_registers
        self.rng = rng or random.Random()
        self.input_keys = input_keys or DEFAULT_INPUT_KEYS

    def _rand_reg(self, low: int = 0) -> int:
        return self.rng.randint(low, self.num_registers - 1)

    def _create_random_instruction(self, kind: str) -> Instruction:
        if kind == "ARITH":
            dest = self._rand_reg(low=3)
            src1 = self._rand_reg(low=0)
            if self.rng.random() < 0.5:
                src2 = self._rand_reg(low=0)
                src2_is_const = False
            else:
                src2 = self.rng.uniform(-5.0, 5.0)
                src2_is_const = True
            op = self.rng.choice(["+", "-", "*", "/"])
            return ArithmeticInstruction(dest=dest, op=op, src1=src1, src2=src2, src2_is_const=src2_is_const)

        if kind == "LOAD_INPUT":
            dest = self._rand_reg(low=0)
            key = self.rng.choice(self.input_keys)
            return LoadInputInstruction(dest=dest, input_key=key)

        if kind == "LOAD_CONST":
            dest = self._rand_reg(low=3)
            value = self.rng.uniform(0.0, 10.0)
            return LoadConstInstruction(dest=dest, value=value)

        if kind == "COPY_REG":
            src = self._rand_reg(low=0)
            dest = self._rand_reg(low=3)
            return CopyRegInstruction(dest=dest, src=src)

        if kind == "CLAMP_MIN":
            src = self._rand_reg(low=0)
            dest = src
            minimum = 0.0
            return ClampMinInstruction(dest=dest, src=src, minimum=minimum)

        if kind == "CLAMP_MAX":
            src = self._rand_reg(low=0)
            dest = src
            maximum = 2.0
            return ClampMaxInstruction(dest=dest, src=src, maximum=maximum)

        if kind == "COND_SKIP":
            cond_reg = self._rand_reg(low=0)
            threshold = self.rng.uniform(0.0, 50.0)
            skip_count = self.rng.randint(1, 3)
            comparison = self.rng.choice([">", "<"])
            return ConditionalSkipInstruction(
                cond_reg=cond_reg,
                threshold=threshold,
                skip_count=skip_count,
                comparison=comparison,
            )

        if kind == "SET_DR":
            reg = self._rand_reg(low=3)
            return SetPortfolioInstruction(component="DR", reg_name=reg)

        if kind == "SET_MH":
            reg_name = self._rand_reg(low=3)
            reg_weight = self._rand_reg(low=3)
            return SetPortfolioInstruction(component="MH1", reg_name=reg_name, reg_weight=reg_weight)

        raise ValueError(f"Unknown instruction kind: {kind}")

    def generate_random_program(self) -> LGPProgram:
        """
        Sinh 1 program random và đảm bảo có ít nhất 1 DR + MH1..MHn.
        """
        length = self.rng.randint(self.min_length, self.max_length)
        instructions: List[Instruction] = []

        for _ in range(length):
            kind = self.rng.choices(
                population=["ARITH", "LOAD_INPUT", "LOAD_CONST", "COPY_REG", "CLAMP_MIN", "CLAMP_MAX", "COND_SKIP"],
                weights=[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1],
            )[0]
            instructions.append(self._create_random_instruction(kind))

        # Đảm bảo DR + MH1..MHn
        base_reg_for_dr = self._rand_reg(low=3)
        instructions.append(SetPortfolioInstruction(component="DR", reg_name=base_reg_for_dr))

        for i in range(LGPConfig.n_mh_genes):
            reg_name = self._rand_reg(low=3)
            reg_weight = self._rand_reg(low=3)
            component = f"MH{i+1}"
            instructions.append(SetPortfolioInstruction(component=component, reg_name=reg_name, reg_weight=reg_weight))

        return LGPProgram(instructions=instructions, num_registers=self.num_registers)
