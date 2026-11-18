"""Check the law the model hands in against fresh drops.

finish() asks for the law as a Python expression for the fall time in terms
of the knob setting h. It is parsed with a whitelist (numbers, h, arithmetic
and functions from the math module; nothing else), then compared with new
timings at heights the model never chose. The timings come from the
apparatus, so the same check works on a real rig.

The constant the model reports can be any of several equivalent ones (g,
g/2, 2/g, sqrt(2/g)...), so it is not compared directly. Instead the g its
law implies is worked out: g = 2h / t(h)^2 at a spread of heights. If the
law has the right form, that number is the same at every height and should
be close to 9.8.
"""

from __future__ import annotations

import ast
import math
import random
import re
import statistics
from types import SimpleNamespace
from typing import Any, Callable

from .apparatus import Apparatus

_MATH_NAMES = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
_EXTRA_FUNCS = {"abs": abs, "min": min, "max": max}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Attribute,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.UAdd, ast.USub,
)
_LHS_RE = re.compile(r"^\s*[A-Za-z_]\w*\s*(\([^()]*\))?\s*=(?!=)")
VARIABLES = ("h",)


class FormulaError(ValueError):
    pass


def normalize_formula(text: str) -> str:
    """Accept the usual ways a law gets written down: 't = ...' loses its
    left-hand side, '^' becomes '**', and fences and a final period go."""
    expr = str(text or "")
    previous = None
    while expr != previous:
        previous = expr
        expr = expr.strip().strip("`").rstrip(".")
    if expr.lower().startswith("python"):
        expr = expr[6:].strip()
    expr = _LHS_RE.sub("", expr, count=1).strip()
    return expr.replace("^", "**")


class _FloatConstants(ast.NodeTransformer):
    """Make integer literals floats, so 10**10**10 overflows instead of hanging."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaError(f"Only numbers are allowed as constants, not {node.value!r}.")
        return ast.copy_location(ast.Constant(float(node.value)), node)


def compile_formula(text: str, variables: tuple[str, ...] = VARIABLES) -> Callable[..., float]:
    expr = normalize_formula(text)
    if not expr:
        raise FormulaError("No formula was given.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Not a valid expression: {exc.msg}") from None
    allowed_names = set(variables) | set(_MATH_NAMES) | set(_EXTRA_FUNCS) | {"math"}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise FormulaError(f"{type(node).__name__} is not allowed in a formula.")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise FormulaError(f"Unknown name {node.id!r}; use {', '.join(variables)} "
                               "and math functions.")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"
                    and node.attr in _MATH_NAMES):
                raise FormulaError("Only math.<function> attribute access is allowed.")
        if isinstance(node, ast.Call) and node.keywords:
            raise FormulaError("Keyword arguments are not allowed in a formula.")
    tree = ast.fix_missing_locations(_FloatConstants().visit(tree))
    code = compile(tree, "<formula>", "eval")
    namespace: dict[str, Any] = {"__builtins__": {}, **_MATH_NAMES, **_EXTRA_FUNCS,
                                 "math": SimpleNamespace(**_MATH_NAMES)}

    def evaluate(*values: float) -> float:
        scope = dict(namespace)
        scope.update(zip(variables, (float(v) for v in values)))
        return float(eval(code, scope))  # noqa: S307 - whitelisted AST, no builtins

    return evaluate


def collect_holdout(apparatus: Apparatus, *, points: int = 30, samples: int = 3,
                    seed: int | None = None) -> list[dict[str, float]]:
    """Fresh timings at random heights the model never chose, through the
    same interface the model used. The knob is put back afterwards."""
    rng = random.Random(seed)
    before = apparatus.get_height()
    holdout = []
    try:
        for _ in range(points):
            h = round(rng.uniform(apparatus.min_height, apparatus.max_height), 3)
            apparatus.set_height(h)
            t = statistics.fmean(apparatus.drop() for _ in range(samples))
            holdout.append({"h": h, "t": t})
    finally:
        apparatus.set_height(before)
    return holdout


def implied_g(law: Callable[..., float], heights: list[float]) -> dict[str, float] | None:
    """The g the law implies, 2h/t(h)^2, at each height: its median and how
    much it varies (0 for a law of the right form)."""
    values = []
    for h in heights:
        try:
            t = law(h)
        except (ArithmeticError, ValueError, TypeError):
            return None
        if not math.isfinite(t) or t <= 0:
            return None
        values.append(2.0 * h / (t * t))
    median = statistics.median(values)
    return {"median": median, "spread": (max(values) - min(values)) / median if median else None}


def score_formula(formula: str, holdout: list[dict[str, float]], *,
                  threshold: float = 0.05) -> dict[str, Any]:
    """Compare the law with the fresh timings. nrmse is the RMS error divided
    by the range of the timings; at or below threshold the law counts as found."""
    result: dict[str, Any] = {"formula": formula, "threshold": threshold, "discovered": False}
    try:
        law = compile_formula(formula)
    except FormulaError as exc:
        result["error"] = str(exc)
        return result
    errors = []
    try:
        for point in holdout:
            errors.append(law(point["h"]) - point["t"])
    except (ArithmeticError, ValueError, TypeError) as exc:
        result["error"] = f"The formula failed to evaluate: {exc}"
        return result
    if not all(math.isfinite(e) for e in errors):
        result["error"] = "The formula gave a non-finite value."
        return result
    times = [p["t"] for p in holdout]
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    nrmse = rms / ((max(times) - min(times)) or 1.0)
    heights = sorted(p["h"] for p in holdout)
    g = implied_g(law, heights)
    result.update({"rmse_s": rms, "nrmse": nrmse, "discovered": nrmse <= threshold,
                   "implied_g": g["median"] if g else None,
                   "implied_g_spread": g["spread"] if g else None,
                   "points": len(holdout)})
    return result
