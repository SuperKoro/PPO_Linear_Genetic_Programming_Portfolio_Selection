from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

Number = float


class Instruction(ABC):
    """
    Base class cho mọi LGP instruction.
    execute() trả về int = số instruction *thêm* cần skip.
    0 nghĩa là chỉ sang instruction tiếp theo.
    """
    @abstractmethod
    def execute(self, registers: List[float], ctx: Dict[str, Any]) -> int:
        ...

    @abstractmethod
    def clone(self) -> "Instruction":
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Instruction":
        ...


@dataclass
class LoadInputInstruction(Instruction):
    dest: int
    input_key: str

    def execute(self, registers: List[float], ctx: Dict[str, Any]) -> int:
        inputs = ctx.get("inputs", {})
        registers[self.dest] = float(inputs.get(self.input_key, 0.0))
        return 0

    def clone(self) -> "LoadInputInstruction":
        return LoadInputInstruction(dest=self.dest, input_key=self.input_key)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "LOAD_INPUT", "dest": self.dest, "input_key": self.input_key}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoadInputInstruction":
        return cls(dest=int(data["dest"]), input_key=str(data["input_key"]))


@dataclass
class LoadConstInstruction(Instruction):
    dest: int
    value: float

    def execute(self, registers: List[float], ctx: Dict[str, Any]) -> int:
        registers[self.dest] = float(self.value)
        return 0

    def clone(self) -> "LoadConstInstruction":
        return LoadConstInstruction(dest=self.dest, value=self.value)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "LOAD_CONST", "dest": self.dest, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoadConstInstruction":
        return cls(dest=int(data["dest"]), value=float(data["value"]))


@dataclass
class CopyRegInstruction(Instruction):
    dest: int
    src: int

    def execute(self, registers: List[float], ctx: Dict[str, Any]) -> int:
        registers[self.dest] = float(registers[self.src])
        return 0

    def clone(self) -> "CopyRegInstruction":
        return CopyRegInstruction(dest=self.dest, src=self.src)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "COPY_REG", "dest": self.dest, "src": self.src}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CopyRegInstruction":
        return cls(dest=int(data["dest"]), src=int(data["src"]))


@dataclass
class ArithmeticInstruction(Instruction):
    dest: int
    op: str   # "+", "-", "*", "/"
    src1: int
    # src2 có thể là index register hoặc hằng số
    src2: Union[int, float]
    src2_is_const: bool = False

    def execute(self, registers: List[float], ctx: Dict[str, Any]) -> int:
        a = float(registers[self.src1])
        if self.src2_is_const:
            b = float(self.src2)
        else:
            b = float(registers[int(self.src2)])
        if self.op == "+":
            registers[self.dest] = a + b
        elif self.op == "-":
            registers[self.dest] = a - b
        elif self.op == "*":
            registers[self.dest] = a * b
        elif self.op == "/":
            if abs(b) < 1e-8:
                registers[self.dest] = a
            else:
                registers[self.dest] = a / b
        return 0

    def clone(self) -> "ArithmeticInstruction":
        return ArithmeticInstruction(
            dest=self.dest,
            op=self.op,
            src1=self.src1,
            src2=self.src2,
            src2_is_const=self.src2_is_const,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ARITH",
            "dest": self.dest,
            "op": self.op,
            "src1": self.src1,
            "src2": self.src2,
            "src2_is_const": self.src2_is_const,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArithmeticInstruction":
        return cls(
            dest=int(data["dest"]),
            op=str(data["op"]),
            src1=int(data["src1"]),
            src2=data["src2"],
            src2_is_const=bool(data.get("src2_is_const", False)),
        )


@dataclass
class ClampMinInstruction(Instruction):
    dest: int
    src: int
    minimum: float

    def execute(self, registers: List[float], ctx: Dict[str, Any]) -> int:
        registers[self.dest] = max(float(registers[self.src]), float(self.minimum))
        return 0

    def clone(self) -> "ClampMinInstruction":
        return ClampMinInstruction(dest=self.dest, src=self.src, minimum=self.minimum)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CLAMP_MIN", "dest": self.dest, "src": self.src, "minimum": self.minimum}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClampMinInstruction":
        return cls(dest=int(data["dest"]), src=int(data["src"]), minimum=float(data["minimum"]))


@dataclass
class ClampMaxInstruction(Instruction):
    dest: int
    src: int
    maximum: float

    def execute(self, registers: List[float], ctx: Dict[str, Any]) -> int:
        registers[self.dest] = min(float(registers[self.src]), float(self.maximum))
        return 0

    def clone(self) -> "ClampMaxInstruction":
        return ClampMaxInstruction(dest=self.dest, src=self.src, maximum=self.maximum)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CLAMP_MAX", "dest": self.dest, "src": self.src, "maximum": self.maximum}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClampMaxInstruction":
        return cls(dest=int(data["dest"]), src=int(data["src"]), maximum=float(data["maximum"]))


@dataclass
class ConditionalSkipInstruction(Instruction):
    cond_reg: int
    threshold: float
    skip_count: int
    comparison: str = ">"  # ">", "<", ">=", "<=", "=="

    def _cond(self, value: float) -> bool:
        if self.comparison == ">":
            return value > self.threshold
        elif self.comparison == "<":
            return value < self.threshold
        elif self.comparison == ">=":
            return value >= self.threshold
        elif self.comparison == "<=":
            return value <= self.threshold
        elif self.comparison == "==":
            return value == self.threshold
        return False

    def execute(self, registers: List[float], ctx: Dict[str, Any]) -> int:
        v = float(registers[self.cond_reg])
        if self._cond(v):
            return int(max(0, self.skip_count))
        return 0

    def clone(self) -> "ConditionalSkipInstruction":
        return ConditionalSkipInstruction(
            cond_reg=self.cond_reg,
            threshold=self.threshold,
            skip_count=self.skip_count,
            comparison=self.comparison,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "COND_SKIP",
            "cond_reg": self.cond_reg,
            "threshold": self.threshold,
            "skip_count": self.skip_count,
            "comparison": self.comparison,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalSkipInstruction":
        return cls(
            cond_reg=int(data["cond_reg"]),
            threshold=float(data["threshold"]),
            skip_count=int(data["skip_count"]),
            comparison=str(data.get("comparison", ">")),
        )


# === Portfolio instruction ===================================================

@dataclass
class SetPortfolioInstruction(Instruction):
    """
    component: "DR", "MH1", "MH2", "MH3"
    reg_name: register index chứa code DR/MH (float)
    reg_weight: optional register index chứa weight (cho MH)
    """
    component: str
    reg_name: int
    reg_weight: Optional[int] = None

    def execute(self, registers: List[float], ctx: Dict[str, Any]) -> int:
        builder = ctx["builder"]
        available_dr = ctx.get("available_dr") or []
        available_mh = ctx.get("available_mh") or []

        def decode_index(x: float, names: List[str]) -> str:
            if not names:
                return ""
            idx = int(abs(x)) % len(names)
            return names[idx]

        if self.component == "DR":
            code = float(registers[self.reg_name])
            dr_name = decode_index(code, available_dr)
            if dr_name:
                builder.set_dr(dr_name)
        else:
            code = float(registers[self.reg_name])
            mh_name = decode_index(code, available_mh)
            if mh_name:
                if self.reg_weight is not None:
                    w = float(registers[self.reg_weight])
                else:
                    w = 1.0
                builder.set_mh(self.component, mh_name, w)

        return 0

    def clone(self) -> "SetPortfolioInstruction":
        return SetPortfolioInstruction(
            component=self.component,
            reg_name=self.reg_name,
            reg_weight=self.reg_weight,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SET_PORTFOLIO",
            "component": self.component,
            "reg_name": self.reg_name,
            "reg_weight": self.reg_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SetPortfolioInstruction":
        return cls(
            component=str(data["component"]),
            reg_name=int(data["reg_name"]),
            reg_weight=data.get("reg_weight"),
        )


# ---- Factory ---------------------------------------------------------------

INSTRUCTION_TYPE_MAP = {
    "LOAD_INPUT": LoadInputInstruction,
    "LOAD_CONST": LoadConstInstruction,
    "COPY_REG": CopyRegInstruction,
    "ARITH": ArithmeticInstruction,
    "CLAMP_MIN": ClampMinInstruction,
    "CLAMP_MAX": ClampMaxInstruction,
    "COND_SKIP": ConditionalSkipInstruction,
    "SET_PORTFOLIO": SetPortfolioInstruction,
}


def instruction_from_dict(data: Dict[str, Any]) -> Instruction:
    t = data.get("type")
    cls = INSTRUCTION_TYPE_MAP.get(t)
    if cls is None:
        raise ValueError(f"Unknown instruction type: {t}")
    return cls.from_dict(data)
