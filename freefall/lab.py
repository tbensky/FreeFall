"""The conversation loop, and where each run's files go.

One session: send the task, let the model call tools until it calls finish
(or runs out of calls), then check its law against fresh drops.

    Runs/<model>/001/transcript.jsonl   every message, tool call and result
                     console.log        what was printed on the terminal
                     measurements.csv   every drop: h and t
                     notebook.md        what it wrote down (and .jsonl)
                     plots/             each graph it made (plots.jsonl has the data)
                     summary.json       outcome, its law and constant, the check
    Runs/<model>/runs_summary.json      how many runs found the law
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .apparatus import Apparatus
from .evaluate import collect_holdout, score_formula
from .llm import ChatClient, LLMError, extract_text_tool_calls
from .notebook import Notebook
from .tools import LabState, ToolResult, Toolbox, default_tools

NUDGE = ("Please carry on by using your tools. When you are satisfied with "
         "your findings, call finish.")
# Sent once, when warn_calls_left model calls remain. {left} is filled in.
# Overridden by prompts/lab_period.txt when the command line finds one.
DEFAULT_LAB_PERIOD_WARNING = (
    "The lab period is almost over: you have {left} turns left. Finish your curve fit, "
    "record your constant in your notebook, and call finish."
)

# Tokens are estimated from characters until the host reports a real count;
# then the ratio it implies is used. Ollama's count can leave out cached
# tokens, so an implausible ratio is ignored.
DEFAULT_CHARS_PER_TOKEN = 3.5
PLAUSIBLE_CHARS_PER_TOKEN = (2.0, 6.0)


@dataclass
class LabConfig:
    max_calls: int = 200          # model calls before the session is stopped
    warn_calls_left: int = 20     # tell the model once when this many are left (0 = never)
    lab_period_warning: str = DEFAULT_LAB_PERIOD_WARNING
    # The model's context window: each call sends the system prompt, the task
    # and as many of the latest exchanges as fit in this many tokens (0 = the
    # whole conversation). Older exchanges are dropped without a word, so
    # older results survive only in the notebook, as for a student.
    context_tokens: int = 8000
    max_idle: int = 5             # replies in a row without a tool call before stopping
    holdout_points: int = 30      # fresh drops used to check the final law
    holdout_samples: int = 3
    threshold: float = 0.05       # normalized RMS error that counts as found
    verbose: bool = True


def _chars(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False))


def trim_to_budget(messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                   budget_tokens: int, chars_per_token: float) -> list[dict[str, Any]]:
    """System prompt + task + the most recent exchanges that fit in the budget.

    The tool definitions go with every call, so they count too. Cuts fall
    only where an exchange starts (an assistant message), so a tool result is
    never separated from its call, and the latest exchange is always kept,
    even if it alone is over the budget (a long read_notebook, say).
    """
    if budget_tokens <= 0 or len(messages) <= 2:
        return messages
    limit = budget_tokens * chars_per_token
    head, rest = messages[:2], messages[2:]
    fixed = _chars(head) + _chars(tools)
    sizes = [_chars(m) for m in rest]
    if fixed + sum(sizes) <= limit:
        return messages
    tail = [0] * (len(rest) + 1)
    for i in range(len(rest) - 1, -1, -1):
        tail[i] = tail[i + 1] + sizes[i]
    starts = [i for i, m in enumerate(rest) if m.get("role") == "assistant"]
    if not starts:
        return messages
    cut = next((s for s in starts if fixed + tail[s] <= limit), starts[-1])
    return messages if cut == 0 else head + rest[cut:]


class _Transcript:
    def __init__(self, folder: Path | None):
        self.path = folder / "transcript.jsonl" if folder else None

    def write(self, **record: Any) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _short(text: str, limit: int = 400) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."


def run_session(client: ChatClient, apparatus: Apparatus, *, system_prompt: str,
                task_prompt: str, folder: Path | None = None,
                config: LabConfig | None = None, holdout_seed: int | None = None,
                extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one session and return (and save) its summary."""
    config = config or LabConfig()
    say: Callable[[str], None] = print if config.verbose else (lambda _msg: None)
    notebook = Notebook(folder)
    state = LabState(apparatus=apparatus, notebook=notebook, folder=folder)
    toolbox = Toolbox(default_tools(apparatus))
    transcript = _Transcript(folder)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt},
                                      {"role": "user", "content": task_prompt}]
    transcript.write(call=0, role="system", content=system_prompt)
    transcript.write(call=0, role="user", content=task_prompt)
    say(f"[task] {_short(task_prompt, 300)}")

    started = time.monotonic()
    status, error = "call_limit", None
    finish_data: dict[str, Any] = {}
    tool_counts: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    idle = text_fallbacks = calls_made = 0
    warned_at: int | None = None
    chars_per_token = DEFAULT_CHARS_PER_TOKEN
    context = {"budget_tokens": config.context_tokens, "trimmed_calls": 0,
               "over_budget_calls": 0, "max_prompt_tokens": 0}

    for call_number in range(1, config.max_calls + 1):
        state.call = calls_made = call_number
        left = config.max_calls - call_number + 1
        if warned_at is None and 0 < config.warn_calls_left == left < config.max_calls:
            warned_at = call_number
            warning = config.lab_period_warning.replace("{left}", str(left))
            messages.append({"role": "user", "content": warning})
            transcript.write(call=call_number, role="user", content=warning)
            say(f"[time] {warning}")
        schemas = toolbox.schemas()
        sent = trim_to_budget(messages, schemas, config.context_tokens, chars_per_token)
        try:
            reply = client.chat(sent, schemas)
        except LLMError as exc:
            status, error = "error", str(exc)
            say(f"[error] {exc}")
            break
        usage.update(reply.usage)
        context["trimmed_calls"] += int(sent is not messages)
        prompt_tokens = int(reply.usage.get("prompt_tokens") or 0)
        if prompt_tokens > 0:
            context["max_prompt_tokens"] = max(context["max_prompt_tokens"], prompt_tokens)
            context["over_budget_calls"] += int(0 < config.context_tokens < prompt_tokens)
            ratio = (_chars(sent) + _chars(schemas)) / prompt_tokens
            if PLAUSIBLE_CHARS_PER_TOKEN[0] <= ratio <= PLAUSIBLE_CHARS_PER_TOKEN[1]:
                chars_per_token = ratio

        calls = list(reply.tool_calls)
        from_text = False
        if not calls and reply.text:
            calls = extract_text_tool_calls(reply.text, schemas)
            from_text = bool(calls)
            text_fallbacks += int(from_text)
        messages.append(reply.assistant_message)
        transcript.write(call=call_number, role="assistant", content=reply.text,
                         tool_calls=[asdict(c) for c in calls], from_text=from_text,
                         usage=reply.usage)
        if reply.text.strip():
            say(f"[{call_number}] model: {_short(reply.text)}")
        if not calls:
            idle += 1
            if idle >= config.max_idle:
                status = "no_tool_calls"
                say(f"[stopped] {idle} replies in a row without a tool call")
                break
            messages.append({"role": "user", "content": NUDGE})
            transcript.write(call=call_number, role="user", content=NUDGE)
            continue
        idle = 0

        text_results, finished = [], False
        for tool_call in calls:
            if tool_call.arguments is None:
                result = ToolResult(f"The arguments for {tool_call.name} could not be read "
                                    f"({tool_call.parse_error}). Send them as a JSON object.")
            else:
                result = toolbox.run(state, tool_call.name, tool_call.arguments)
            tool_counts[tool_call.name] += 1
            transcript.write(call=call_number, role="tool", name=tool_call.name,
                             arguments=tool_call.arguments, content=result.text,
                             data=result.data)
            say(f"[{call_number}] {tool_call.name}({json.dumps(tool_call.arguments)}) -> "
                f"{_short(result.text, 200)}")
            if from_text:
                text_results.append(f"Result of {tool_call.name}: {result.text}")
            else:
                messages.append(client.tool_result_message(tool_call, result.text))
            if result.finished:
                finished, finish_data = True, result.data
                break
        if text_results:
            messages.append({"role": "user", "content": "\n".join(text_results)})
        if finished:
            status = "finished"
            break

    evaluation: dict[str, Any] | None = None
    if status == "finished":
        say("[check] testing the law against fresh drops...")
        try:
            holdout = collect_holdout(apparatus, points=config.holdout_points,
                                      samples=config.holdout_samples, seed=holdout_seed)
            evaluation = score_formula(finish_data.get("formula", ""), holdout,
                                       threshold=config.threshold)
        except Exception as exc:  # noqa: BLE001 - a failed check must not lose the run
            evaluation = {"error": f"{type(exc).__name__}: {exc}", "discovered": False}

    summary: dict[str, Any] = {
        "status": status,
        "discovered": bool(evaluation and evaluation.get("discovered")),
        "conclusion": finish_data.get("conclusion"),
        "formula": finish_data.get("formula"),
        "constant": finish_data.get("constant"),
        "nrmse": evaluation.get("nrmse") if evaluation else None,
        # the g its law implies (2h/t^2), whatever constant it chose to report
        "implied_g": evaluation.get("implied_g") if evaluation else None,
        "model_calls": calls_made,
        "tool_counts": dict(tool_counts),
        "drops": state.drops,
        "knob_moves": state.knob_moves,
        "distinct_heights": len({m["h"] for m in state.measurements}),
        "notebook_entries": len(notebook.entries),
        "plots": len(state.plots),
        "plot_files": [p["file"] for p in state.plots if p.get("file")],
        "lab_period_warning_call": warned_at,
        "context": {**context, "chars_per_token": round(chars_per_token, 2)},
        "text_tool_calls": text_fallbacks,
        "usage": dict(usage),
        "elapsed_s": round(time.monotonic() - started, 1),
        "error": error,
        **(extra or {}),
        "evaluation": evaluation,
        # exactly what the model was told about each tool
        "tool_definitions": [s["function"] for s in toolbox.schemas()],
    }
    if folder is not None:
        (folder / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_jsonable) + "\n",
            encoding="utf-8")
    verdict = "found the law" if summary["discovered"] else "did not find the law"
    if status != "finished":
        verdict = f"stopped ({status})"
    detail = ""
    if isinstance(summary["nrmse"], float):
        detail += f", nrmse={summary['nrmse']:.3f}"
    if isinstance(summary["implied_g"], float):
        detail += f", implied g={summary['implied_g']:.3f}"
    say(f"[result] {verdict}{detail}; formula: {summary['formula']!r}; "
        f"constant: {summary['constant']!r}")
    return summary


def _jsonable(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else str(value)


# --------------------------------------------------------------------------
# Run folders: Runs/<model>/001, 002, ... and Runs/<model>/runs_summary.json
# --------------------------------------------------------------------------
RUN_DIR_RE = re.compile(r"^\d{3,}$")
NO_VERDICT = {"error", "interrupted", "missing"}


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def numbered_run_dirs(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        return []
    return sorted((p for p in model_dir.iterdir() if p.is_dir() and RUN_DIR_RE.match(p.name)),
                  key=lambda p: int(p.name))


def next_run_dir(model_dir: Path) -> tuple[int, Path]:
    """Create the next numbered folder; numbering continues across batches."""
    model_dir.mkdir(parents=True, exist_ok=True)
    number = max((int(p.name) for p in numbered_run_dirs(model_dir)), default=0) + 1
    while True:
        run_dir = model_dir / f"{number:03d}"
        try:
            run_dir.mkdir()
            return number, run_dir
        except FileExistsError:
            number += 1


def write_runs_summary(model_dir: Path, model: str) -> dict[str, Any]:
    """Rebuild runs_summary.json from every numbered run folder."""
    rows = []
    for run_dir in numbered_run_dirs(model_dir):
        try:
            s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            s = {"status": "missing"}
        rows.append({key: s.get(key) for key in (
            "status", "discovered", "formula", "constant", "nrmse", "implied_g", "model_calls",
            "drops", "distinct_heights", "notebook_entries", "plots", "seed", "error")}
            | {"run": run_dir.name})
    judged = [r for r in rows if r["status"] not in NO_VERDICT]
    found = [r for r in rows if r["discovered"]]
    gs = [r["implied_g"] for r in found if isinstance(r["implied_g"], (int, float))]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    summary = {
        "model": model,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": len(rows),
        "judged": len(judged),
        "discovered": len(found),
        "discovery_rate": round(len(found) / len(judged), 3) if judged else None,
        "median_implied_g_when_discovered": statistics.median(gs) if gs else None,
        "statuses": dict(sorted(statuses.items())),
        "run_list": rows,
    }
    (model_dir / "runs_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
