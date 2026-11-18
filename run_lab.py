#!/usr/bin/env python3
"""Let an LLM loose on the falling-ball experiment and see if it finds the law.

Examples:
    # a hosted model on NRP (same LLM_* settings as GASolver and Malus)
    export LLM_API_KEY=...            # your NRP key
    python3 run_lab.py --provider nrp --model deepseek-v4-flash

    # a local Ollama model, 10 independent runs
    python3 run_lab.py --provider ollama --model qwen3:8b --runs 10

Results go in Runs/<model>/<NNN>/, with Runs/<model>/runs_summary.json
counting how many runs found the law.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from freefall.apparatus import DropConfig, HardwareApparatus, SimulatedDropApparatus  # noqa: E402
from freefall.lab import (DEFAULT_LAB_PERIOD_WARNING, LabConfig, next_run_dir,  # noqa: E402
                          run_session, safe_model_name, write_runs_summary)
from freefall.llm import make_client  # noqa: E402

MAX_CONSECUTIVE_ERRORS = 3


class _Tee:
    """Passes everything written to the terminal through, and into a file."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def console_log(path: Path):
    """Copy stdout into path while the block runs (line-buffered)."""
    previous = sys.stdout
    with path.open("a", encoding="utf-8", buffering=1) as file:
        sys.stdout = _Tee(previous, file)
        try:
            yield
        finally:
            sys.stdout = previous


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    llm = p.add_argument_group("model")
    llm.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "nrp"),
                     choices=("nrp", "openai", "deepseek", "ollama"),
                     help="nrp/openai/deepseek: OpenAI-compatible host; ollama: local /api/chat")
    llm.add_argument("--llm-url", default=os.environ.get("LLM_URL"),
                     help="Endpoint URL (default depends on --provider)")
    llm.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
    llm.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", ""),
                     help="API key for a hosted model (default: $LLM_API_KEY)")
    llm.add_argument("--temperature", type=float, default=None,
                     help="Sampling temperature (default: the host's own)")
    llm.add_argument("--num-ctx", type=int, default=16384,
                     help="Ollama's own context window; needs room for --context-tokens plus "
                          "the reply")
    llm.add_argument("--think", choices=("on", "off"), default=None,
                     help="Ollama only: turn a thinking model's thinking on or off")
    llm.add_argument("--timeout", type=float, default=300.0, help="Seconds per model call")
    llm.add_argument("--retries", type=int, default=8,
                     help="Attempts per model call when the host errors (waits 2, 4, 8, 16, "
                          "then 30 s)")

    lab = p.add_argument_group("session")
    lab.add_argument("--runs", type=int, default=1, help="Independent runs to do in a row")
    lab.add_argument("--max-calls", type=int, default=200, help="Model calls per run")
    lab.add_argument("--warn-calls-left", type=int, default=20,
                     help="Tell the model the lab period is nearly over when this many calls "
                          "are left (0 = never)")
    lab.add_argument("--context-tokens", type=int, default=8000,
                     help="The model's context window: each call sends the task and as many "
                          "recent exchanges as fit in this many tokens (0 = everything)")
    lab.add_argument("--threshold", type=float, default=0.05,
                     help="Normalized RMS error on fresh drops that counts as finding the law")
    lab.add_argument("--holdout-points", type=int, default=30,
                     help="Fresh heights used to check the final law")
    lab.add_argument("--system-prompt-file", default=str(HERE / "prompts" / "system.txt"))
    lab.add_argument("--task-prompt-file", default=str(HERE / "prompts" / "task.txt"))
    lab.add_argument("--lab-period-file", default=str(HERE / "prompts" / "lab_period.txt"),
                     help="The warning; {left} is replaced by the calls left")
    lab.add_argument("--out-dir", default=str(HERE / "Runs"))
    lab.add_argument("--quiet", action="store_true", help="Only print one line per run")

    rig = p.add_argument_group("apparatus (simulation)")
    rig.add_argument("--apparatus", choices=("sim", "hardware"), default="sim")
    rig.add_argument("--seed", type=int, default=1,
                     help="Base seed; run NNN uses seed + NNN - 1 for its noise")
    rig.add_argument("--g", type=float, default=9.8, help="Acceleration of the fall, m/s^2")
    rig.add_argument("--noise", type=float, default=0.002,
                     help="Timing noise, seconds (standard deviation)")
    rig.add_argument("--timer-offset", type=float, default=0.0,
                     help="A constant added to every timing, seconds (release latency)")
    return p


def make_apparatus(args: argparse.Namespace, seed: int):
    if args.apparatus == "hardware":
        return HardwareApparatus(), {"apparatus": "hardware"}
    config = DropConfig(g=args.g, noise_s=args.noise, timer_offset_s=args.timer_offset)
    return (SimulatedDropApparatus(config, seed=seed),
            {"apparatus": "simulated", "simulation": asdict(config)})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        print("--runs must be 1 or more", file=sys.stderr)
        return 2
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    task_prompt = Path(args.task_prompt_file).read_text(encoding="utf-8").strip()
    lab_period_path = Path(args.lab_period_file)
    lab_period_warning = (lab_period_path.read_text(encoding="utf-8").strip()
                          if lab_period_path.exists() else DEFAULT_LAB_PERIOD_WARNING)
    think = None if args.think is None else args.think == "on"
    client = make_client(args.provider, args.model, url=args.llm_url,
                         api_key=args.llm_api_key or None, temperature=args.temperature,
                         num_ctx=args.num_ctx, think=think, timeout=args.timeout,
                         attempts=args.retries)
    config = LabConfig(max_calls=args.max_calls, warn_calls_left=args.warn_calls_left,
                       lab_period_warning=lab_period_warning,
                       context_tokens=args.context_tokens, threshold=args.threshold,
                       holdout_points=args.holdout_points, verbose=not args.quiet)
    model_dir = Path(args.out_dir).resolve() / safe_model_name(args.model)
    consecutive_errors = 0

    for index in range(args.runs):
        number, run_dir = next_run_dir(model_dir)
        seed = args.seed + number - 1
        with console_log(run_dir / "console.log"):
            print(f"\n[run {index + 1}/{args.runs}] {run_dir} seed={seed}")
            try:
                apparatus, rig_info = make_apparatus(args, seed)
            except NotImplementedError as exc:
                print(f"[error] {exc}")
                shutil.rmtree(run_dir, ignore_errors=True)
                return 2
            extra = {"model": args.model, "provider": args.provider, "run": f"{number:03d}",
                     "seed": seed, **rig_info,
                     "settings": {"max_calls": args.max_calls,
                                  "warn_calls_left": args.warn_calls_left,
                                  "context_tokens": args.context_tokens,
                                  "threshold": args.threshold, "temperature": args.temperature},
                     "prompts": {"system": system_prompt, "task": task_prompt,
                                 "lab_period": lab_period_warning if args.warn_calls_left > 0
                                 else None}}
            try:
                summary = run_session(client, apparatus, system_prompt=system_prompt,
                                      task_prompt=task_prompt, folder=run_dir, config=config,
                                      holdout_seed=seed + 1_000_000, extra=extra)
            except KeyboardInterrupt:
                (run_dir / "summary.json").write_text(
                    json.dumps({"status": "interrupted", **extra}, indent=2) + "\n",
                    encoding="utf-8")
                write_runs_summary(model_dir, args.model)
                print(f"\n[interrupted] summary so far: {model_dir / 'runs_summary.json'}")
                return 130
            finally:
                apparatus.close()
            totals = write_runs_summary(model_dir, args.model)
            print(f"[run {number:03d}] {summary['status']}, "
                  f"{'found the law' if summary['discovered'] else 'law not found'} | "
                  f"{args.model}: {totals['discovered']} of {totals['judged']} runs found it")
            if summary["status"] == "error":
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print("[stopped] three runs in a row ended in an error; check the endpoint")
                    return 1
            else:
                consecutive_errors = 0
    print(f"[runs summary] {model_dir / 'runs_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
