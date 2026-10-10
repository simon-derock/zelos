"""Safe arithmetic evaluator (no eval/exec).

Inputs : an expression string, e.g. "(0.41 - 0.30) / 0.30 * 100".
Outputs: float. Allowed: numeric literals, + - * / // ** %, unary +/-, parentheses and the
functions abs/min/max/round. Everything else (names, attributes, strings, comprehensions,
other calls, huge exponents) raises CalcError.
"""
from __future__ import annotations

import ast
import math
import operator

MAX_EXP = 64
MAX_ABS = 1e300
MAX_LEN = 300


class CalcError(ValueError):
    pass


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "min": min, "max": max, "round": round}


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CalcError("non-numeric value")
    if isinstance(v, float) and not math.isfinite(v):
        raise CalcError("non-finite result")
    if abs(v) > MAX_ABS:
        raise CalcError("value too large")
    return v


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant):
        return _num(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > MAX_EXP:
            raise CalcError("exponent too large")
        try:
            return _num(_BIN[type(node.op)](a, b))
        except ZeroDivisionError as e:
            raise CalcError("division by zero") from e
        except OverflowError as e:
            raise CalcError("overflow") from e
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _num(_UN[type(node.op)](_ev(node.operand)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise CalcError("function not allowed")
        if not node.args or len(node.args) > 8:
            raise CalcError("bad arguments")
        return _num(_FUNCS[node.func.id](*[_ev(a) for a in node.args]))
    raise CalcError(f"disallowed syntax: {type(node).__name__}")


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression safely; raises CalcError on anything else."""
    if not isinstance(expr, str) or not expr.strip() or len(expr) > MAX_LEN:
        raise CalcError("empty or too long")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise CalcError("syntax error") from e
    return float(_ev(tree))
