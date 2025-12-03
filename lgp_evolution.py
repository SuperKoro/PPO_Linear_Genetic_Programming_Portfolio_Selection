from __future__ import annotations
from typing import List
import numpy as np
import random

from config import LGPConfig
from lgp_program import LGPProgram
from lgp_generator import LGPGenerator


def linear_crossover(
    parent1: LGPProgram,
    parent2: LGPProgram,
    rng: random.Random,
    max_length: int | None = None,
) -> LGPProgram:
    """
    Two-point crossover trên chuỗi instruction.
    """
    inst1 = parent1.instructions
    inst2 = parent2.instructions
    len1 = len(inst1)
    len2 = len(inst2)
    if len1 == 0 or len2 == 0:
        return parent1.clone() if rng.random() < 0.5 else parent2.clone()

    p1_start = rng.randint(0, len1 - 1)
    p1_end = rng.randint(p1_start + 1, len1)

    p2_start = rng.randint(0, len2 - 1)
    p2_end = rng.randint(p2_start + 1, len2)

    child_instructions = []
    child_instructions.extend(inst1[:p1_start])
    child_instructions.extend([ins.clone() for ins in inst2[p2_start:p2_end]])
    child_instructions.extend([ins.clone() for ins in inst1[p1_end:]])

    max_len = max_length or LGPConfig.max_program_length
    if len(child_instructions) > max_len:
        child_instructions = child_instructions[:max_len]

    return LGPProgram(instructions=child_instructions, num_registers=parent1.num_registers)


def mutate_program(
    program: LGPProgram,
    generator: LGPGenerator,
    rng: random.Random,
    mutation_rate: float,
) -> LGPProgram:
    """
    Gồm các mutation: MODIFY / INSERT / DELETE / SWAP.
    """
    if rng.random() >= mutation_rate:
        return program

    mutated = program.clone()
    n = len(mutated.instructions)
    if n == 0:
        return mutated

    mutation_types = ["MODIFY", "INSERT", "DELETE", "SWAP"]
    mtype = rng.choices(mutation_types, weights=[0.4, 0.3, 0.2, 0.1])[0]

    if mtype == "MODIFY":
        idx = rng.randrange(n)
        kind = rng.choice(["ARITH", "LOAD_INPUT", "LOAD_CONST", "COPY_REG", "CLAMP_MIN", "CLAMP_MAX", "COND_SKIP"])
        mutated.instructions[idx] = generator._create_random_instruction(kind)

    elif mtype == "INSERT":
        if n < LGPConfig.max_program_length:
            idx = rng.randrange(n + 1)
            kind = rng.choice(["ARITH", "LOAD_INPUT", "LOAD_CONST", "COPY_REG", "COND_SKIP"])
            mutated.instructions.insert(idx, generator._create_random_instruction(kind))

    elif mtype == "DELETE":
        if n > LGPConfig.min_program_length:
            idx = rng.randrange(n)
            mutated.instructions.pop(idx)

    elif mtype == "SWAP":
        if n > 1:
            i = rng.randrange(n)
            j = rng.randrange(n)
            mutated.instructions[i], mutated.instructions[j] = mutated.instructions[j], mutated.instructions[i]

    return mutated
