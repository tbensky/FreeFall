"""The tools the model can call.

Each tool is a small class with a name, a description, a JSON schema for its
arguments and a run() method. They reach the rig only through the Apparatus
interface, so they work unchanged on the simulation and on real hardware.
To add a tool, write a class like these and add it to default_tools().

The descriptions are plain on purpose: a knob, a button, a time. They are
what the model reads.
"""

from __future__ import annotations

import csv
import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .apparatus import Apparatus, ApparatusError
from .notebook import Notebook


@dataclass
class ToolResult:
    text: str                      # what the model is shown
    finished: bool = False         # the model announced it is done
    data: dict[str, Any] = field(default_factory=dict)  # for the logs, not the model


@dataclass
class LabState:
    """Shared by all tools during one session."""

    apparatus: Apparatus
    notebook: Notebook
    folder: Path | None = None
    call: int = 0                  # current model call, set by the loop
    drops: int = 0                 # button pushes
    knob_moves: int = 0
    measurements: list[dict[str, Any]] = field(default_factory=list)  # every drop
    plots: list[dict[str, Any]] = field(default_factory=list)
    _csv_started: bool = False

    def log_drop(self, h: float, t: float) -> None:
        self.measurements.append({"call": self.call, "h": h, "t": t})
        if self.folder is None:
            return
        with (self.folder / "measurements.csv").open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not self._csv_started:
                writer.writerow(["call", "time", "h", "t"])
                self._csv_started = True
            writer.writerow([self.call, round(time.time(), 3), h, t])


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function schema (Ollama accepts the same shape)."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}

    @abstractmethod
    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        ...


class SetKnob(Tool):
    """Turn the knob. The reply only confirms: keeping track of the setting
    is the model's job."""

    name = "set_knob"

    def __init__(self, apparatus: Apparatus):
        lo, hi = apparatus.min_height, apparatus.max_height
        self.description = f"Turn the knob to a new setting, from {lo:g} to {hi:g}."
        self.parameters = {
            "type": "object",
            "properties": {"h": {"type": "number",
                                 "description": f"The setting, {lo:g} to {hi:g}."}},
            "required": ["h"],
        }

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        if args.get("h") is None:
            return ToolResult("Give the setting to turn the knob to.")
        try:
            state.apparatus.set_height(args["h"])
        except ApparatusError as exc:
            return ToolResult(str(exc), data={"problem": str(exc)})
        state.knob_moves += 1
        return ToolResult("Ok. The knob has been set.",
                          data={"h": state.apparatus.get_height()})


class PushTheButton(Tool):
    """Drop the ball: one timing, and nothing else."""

    name = "push_the_button"
    description = "Push the button."
    parameters = {"type": "object", "properties": {}}

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        app = state.apparatus
        h = app.get_height()
        try:
            t = app.drop()
        except ApparatusError as exc:
            return ToolResult(f"Nothing happened: {exc}")
        state.drops += 1
        state.log_drop(h, t)
        return ToolResult(f"{t:.4f} {app.units}", data={"h": h, "t": t})


class WriteDown(Tool):
    name = "write_down"
    description = (
        "Write something in your lab notebook: data, observations, ideas, plans, anything "
        "you want to keep. The text goes into your lab notebook verbatim, with no "
        "processing, for your own personal benefit. No action will be taken on any text "
        "you write down."
    )
    parameters = {"type": "object",
                  "properties": {"text": {"type": "string", "description": "What to write down."}},
                  "required": ["text"]}

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        text = str(args.get("text") or "").strip()
        if not text:
            return ToolResult("Nothing was written: the text was empty.")
        entry = state.notebook.write(text, call=state.call)
        return ToolResult(f"Written as notebook entry {entry.number}.",
                          data={"entry": entry.number})


class ReadNotebook(Tool):
    name = "read_notebook"
    description = "Read back everything you have written in your lab notebook."
    parameters = {"type": "object", "properties": {}}

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        return ToolResult(state.notebook.read(), data={"entries": len(state.notebook.entries)})


class Plot(Tool):
    """Graph (x, y) points the model supplies, with its own curve, like a
    plot in a spreadsheet. It answers only with a receipt: no curve values,
    differences or RMS, so judging the fit is the model's own work. The
    harness keeps the RMS for the summary."""

    name = "plot"
    MAX_POINTS = 500
    CURVE_POINTS = 241
    description = (
        "Make an X/Y graph from data you supply, and optionally draw your own curve through "
        "it, the way you would make a plot in a spreadsheet. You choose what goes on each "
        "axis and give the points yourself. The graph is saved with your lab work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "points": {"type": "array",
                       "description": ("The data to plot, as a list of [x, y] pairs, for "
                                       "example [[1, 2.3], [2, 4.1]]."),
                       "items": {"type": "array", "items": {"type": "number"},
                                 "minItems": 2, "maxItems": 2}},
            "formula": {"type": "string",
                        "description": ("Optional. Your curve: a Python expression in x with "
                                        "your chosen constants filled in as numbers, for "
                                        "example \"2*x + 1\". Functions from Python's math "
                                        "module may be used by name.")},
            "x_label": {"type": "string", "description": "Optional label for the x axis."},
            "y_label": {"type": "string", "description": "Optional label for the y axis."},
            "title": {"type": "string", "description": "Optional title for the graph."},
        },
        "required": ["points"],
    }

    @staticmethod
    def _read_points(raw: Any) -> tuple[list[tuple[float, float]] | None, str | None]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None, "points must be a list of [x, y] pairs."
        if not isinstance(raw, list) or not raw:
            return None, "points must be a non-empty list of [x, y] pairs."
        points = []
        for item in raw:
            if isinstance(item, dict) and "x" in item and "y" in item:
                item = [item["x"], item["y"]]
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return None, f"Each point must be an [x, y] pair; {item!r} is not."
            try:
                x, y = float(item[0]), float(item[1])
            except (TypeError, ValueError):
                return None, f"Each point must be two numbers; {item!r} is not."
            if not (math.isfinite(x) and math.isfinite(y)):
                return None, f"Each point must be two finite numbers; {item!r} is not."
            points.append((x, y))
        return points, None

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        from .evaluate import FormulaError, compile_formula
        from .plotting import render

        points, problem = self._read_points(args.get("points"))
        if problem:
            return ToolResult(problem)
        if len(points) > self.MAX_POINTS:
            return ToolResult(f"That is {len(points)} points; plot at most {self.MAX_POINTS}.")
        points.sort()
        formula = str(args.get("formula") or "").strip()
        law = None
        if formula:
            try:
                law = compile_formula(formula, ("x",))
            except FormulaError as exc:
                return ToolResult(f"Your formula could not be used: {exc} "
                                  "The curve must be written in terms of x.")
        curve, residuals = None, []
        try:
            if law is not None:
                residuals = [y - law(x) for x, y in points]
                lo, hi = points[0][0], points[-1][0]
                if hi > lo:
                    step = (hi - lo) / (self.CURVE_POINTS - 1)
                    curve = [(lo + i * step, law(lo + i * step)) for i in range(self.CURVE_POINTS)]
        except (ArithmeticError, ValueError, TypeError) as exc:
            return ToolResult(f"Your formula failed to evaluate: {exc}")
        number = len(state.plots) + 1
        x_label = str(args.get("x_label") or "").strip() or "x"
        y_label = str(args.get("y_label") or "").strip() or "y"
        title = str(args.get("title") or "").strip() or f"{y_label} vs {x_label}"
        saved = None
        if state.folder is not None:
            folder = state.folder / "plots"
            folder.mkdir(exist_ok=True)
            saved = render(folder / f"plot_{number:02d}", x_label=x_label, y_label=y_label,
                           title=title, points=points, curve=curve, curve_label=formula)
        rms = math.sqrt(sum(r * r for r in residuals) / len(residuals)) if residuals else None
        record = {"number": number, "call": state.call, "formula": formula, "title": title,
                  "x_label": x_label, "y_label": y_label, "points": len(points), "rms": rms,
                  "file": saved.name if saved else None}
        state.plots.append(record)
        if state.folder is not None:
            with (state.folder / "plots" / "plots.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**record, "data": points}, ensure_ascii=False) + "\n")
        return ToolResult(f"Plot {number} saved.", data=record)


class Finish(Tool):
    name = "finish"
    description = "Announce that you are done, once you are satisfied with your findings."
    parameters = {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string",
                           "description": "Your findings: the law, the constant, and how you know."},
            "formula": {"type": "string",
                        "description": ("Your law as a single Python expression for the time in "
                                        "seconds in terms of the knob setting h, with your "
                                        "constant filled in as a number. Functions from Python's "
                                        "math module may be used by name.")},
            "constant": {"type": "number",
                         "description": "The value of the constant you determined."},
        },
        "required": ["conclusion", "formula", "constant"],
    }

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        return ToolResult("Thank you. The experiment is over.", finished=True,
                          data={"conclusion": str(args.get("conclusion") or "").strip(),
                                "formula": str(args.get("formula") or "").strip(),
                                "constant": args.get("constant")})


def default_tools(apparatus: Apparatus) -> list[Tool]:
    return [SetKnob(apparatus), PushTheButton(), WriteDown(), ReadNotebook(), Plot(), Finish()]


class Toolbox:
    """Looks tools up by name and runs them, turning every failure into text."""

    def __init__(self, tools: list[Tool]):
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def run(self, state: LabState, name: str, args: dict[str, Any] | None) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(f"There is no tool called {name!r}. "
                              f"The tools are: {', '.join(self.tools)}.")
        if not isinstance(args, dict):
            return ToolResult(f"The arguments for {name} could not be read. "
                              "Send them as a JSON object.")
        try:
            return tool.run(state, args)
        except Exception as exc:  # noqa: BLE001 - shown to the model, logged
            return ToolResult(f"{name} failed: {exc}", data={"exception": repr(exc)})


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Tool arguments arrive as a dict (Ollama) or a JSON string (OpenAI)."""
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return None, f"could not parse arguments: {exc}"
    if not isinstance(value, dict):
        return None, "arguments were not a JSON object"
    return value, None
