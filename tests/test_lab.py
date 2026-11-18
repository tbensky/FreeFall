"""Offline tests: the rig, the formula check, and full sessions against a
fake OpenAI-style and a fake Ollama server (no real model, no network)."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from freefall.apparatus import ApparatusError, DropConfig, SimulatedDropApparatus  # noqa: E402
from freefall.evaluate import (FormulaError, collect_holdout, compile_formula,  # noqa: E402
                               score_formula)
from freefall.lab import LabConfig, run_session, trim_to_budget  # noqa: E402
from freefall.llm import make_client  # noqa: E402

LAW = "sqrt(2*h/9.8)"


def quiet_rig(**kw) -> SimulatedDropApparatus:
    return SimulatedDropApparatus(DropConfig(noise_s=0.0, **kw), seed=1)


class ApparatusTests(unittest.TestCase):
    def test_the_fall_time_is_sqrt_2h_over_g(self):
        rig = quiet_rig()
        for h in (0.1, 0.5, 1.0, 2.0):
            rig.set_height(h)
            self.assertAlmostEqual(rig.drop(), math.sqrt(2 * h / 9.8))
        rig = quiet_rig(timer_offset_s=0.01, g=9.0)
        rig.set_height(1.8)
        self.assertAlmostEqual(rig.drop(), math.sqrt(2 * 1.8 / 9.0) + 0.01)

    def test_the_knob_has_limits(self):
        rig = quiet_rig()
        for bad in (0.05, 2.5, "high", float("nan")):
            with self.assertRaises(ApparatusError):
                rig.set_height(bad)
        self.assertEqual(rig.get_height(), 0.1)          # where it starts

    def test_noise_is_reproducible_by_seed(self):
        a, b = (SimulatedDropApparatus(DropConfig(), seed=7) for _ in range(2))
        self.assertEqual([a.drop() for _ in range(5)], [b.drop() for _ in range(5)])
        self.assertNotEqual(len({a.drop() for _ in range(5)}), 1)    # there is noise


class FormulaTests(unittest.TestCase):
    def holdout(self):
        return collect_holdout(quiet_rig(), points=20, samples=1, seed=3)

    def test_the_right_law_passes_in_any_form_and_wrong_ones_fail(self):
        for good in (LAW, "t = 0.4518*h**0.5", "math.sqrt(h/4.9)", "(2*h/9.8)^(1/2)"):
            result = score_formula(good, self.holdout())
            self.assertTrue(result["discovered"], good)
            self.assertAlmostEqual(result["implied_g"], 9.8, delta=0.02)
            self.assertLess(result["implied_g_spread"], 0.01)
        for wrong in ("0.32*h + 0.15", "sqrt(2*h/3)", "h**2", "0.45"):
            self.assertFalse(score_formula(wrong, self.holdout())["discovered"], wrong)

    def test_a_linear_law_implies_a_g_that_changes_with_height(self):
        result = score_formula("0.3*h + 0.15", self.holdout())
        self.assertGreater(result["implied_g_spread"], 0.2)      # not one constant

    def test_unsafe_or_unknown_input_is_rejected(self):
        for bad in ("__import__('os')", "open('x')", "g*h", "", "A + h"):
            with self.assertRaises(FormulaError, msg=bad):
                compile_formula(bad)
        self.assertIn("error", score_formula("sqrt(2*h/g)", self.holdout()))

    def test_the_check_puts_the_knob_back(self):
        rig = quiet_rig()
        rig.set_height(1.3)
        collect_holdout(rig, points=5, samples=1, seed=1)
        self.assertEqual(rig.get_height(), 1.3)


class TrimTests(unittest.TestCase):
    def test_the_budget_keeps_the_task_and_the_latest_exchanges(self):
        from freefall.lab import _chars
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
        for n in range(30):
            messages.append({"role": "assistant", "content": "x" * 200,
                             "tool_calls": [{"id": f"c{n}"}]})
            messages.append({"role": "tool", "tool_call_id": f"c{n}", "content": "r" * 200})
        tools = [{"function": {"name": "t"}}]
        sent = trim_to_budget(messages, tools, budget_tokens=1000, chars_per_token=3.5)
        self.assertEqual(sent[:2], messages[:2])
        self.assertEqual(sent[2]["role"], "assistant")          # no note, cut at an exchange
        self.assertEqual(sent[-1], messages[-1])
        self.assertLessEqual(sum(map(_chars, sent)) + _chars(tools), 3500)
        self.assertIs(trim_to_budget(messages, tools, 0, 3.5), messages)


class FakeModelServer:
    def __init__(self, style: str, script: list):
        self.style, self.script, self.requests, self.problems = style, list(script), [], []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                outer.check(payload)
                step = outer.script.pop(0) if outer.script else ("text", "I am done.")
                body = json.dumps(outer.respond(step)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        path = "/v1/chat/completions" if style == "openai" else "/api/chat"
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}{path}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def check(self, payload):
        names = {t["function"]["name"] for t in payload.get("tools", [])}
        if names != {"set_knob", "push_the_button", "write_down", "read_notebook", "plot",
                     "finish"}:
            self.problems.append(f"tools offered: {sorted(names)}")
        messages = payload["messages"]
        open_ids: list = []
        for m in messages:
            if m["role"] == "assistant":
                open_ids = [c.get("id") for c in m.get("tool_calls") or []]
            elif m["role"] == "tool":
                if self.style == "openai" and m.get("tool_call_id") not in open_ids:
                    self.problems.append(f"orphan tool result {m.get('tool_call_id')}")
                if self.style == "ollama" and not m.get("tool_name"):
                    self.problems.append("ollama tool result without tool_name")

    def respond(self, step):
        kind, value = step
        if kind == "text":
            message = {"role": "assistant", "content": value}
        else:   # ("tools", [(name, args), ...])
            calls = []
            for n, (name, args) in enumerate(value):
                function = {"name": name,
                            "arguments": json.dumps(args) if self.style == "openai" else args}
                call = {"function": function}
                if self.style == "openai":
                    call.update({"id": f"call_{len(self.requests)}_{n}", "type": "function"})
                calls.append(call)
            message = {"role": "assistant", "content": "", "tool_calls": calls}
        if self.style == "openai":
            return {"choices": [{"message": message}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        return {"message": message, "prompt_eval_count": 10, "eval_count": 2}



EXPERIMENT = [
    ("tools", [("set_knob", {"h": 0.5}), ("push_the_button", {})]),
    ("tools", [("set_knob", {"h": 2.0}), ("push_the_button", {})]),
    ("tools", [("set_knob", {"h": 3})]),
    ("tools", [("write_down", {"text": "t grows like sqrt(h)"})]),
    ("tools", [("read_notebook", {})]),
    ("tools", [("plot", {"points": [[0.5, 0.32], [2.0, 0.64]], "formula": "0.4518*x**0.5",
                         "x_label": "h (m)", "y_label": "t (s)"})]),
    ("tools", [("finish", {"conclusion": "h = g t^2 / 2, g = 9.8 m/s^2", "formula": LAW,
                           "constant": 9.8})]),
]


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, style, script, **config):
        server = FakeModelServer(style, script)
        self.addCleanup(server.close)
        client = make_client("ollama" if style == "ollama" else "nrp", "fake-model",
                             url=server.url, api_key="test-key")
        summary = run_session(client, SimulatedDropApparatus(seed=1), system_prompt="sys",
                              task_prompt="task", folder=self.folder,
                              config=LabConfig(verbose=False, **config), holdout_seed=5)
        self.assertEqual(server.problems, [])
        return summary, server

    def test_a_full_experiment(self):
        for style in ("openai", "ollama"):
            with self.subTest(style=style):
                self.tmp.cleanup()
                self.tmp = tempfile.TemporaryDirectory()
                self.folder = Path(self.tmp.name)
                summary, server = self.session(style, EXPERIMENT)
                self.assertEqual((summary["status"], summary["discovered"]), ("finished", True))
                self.assertAlmostEqual(summary["implied_g"], 9.8, delta=0.01)
                self.assertEqual(summary["constant"], 9.8)
                self.assertEqual((summary["drops"], summary["distinct_heights"]), (2, 2))
                shown = [m["content"] for m in server.requests[-1]["messages"]
                         if m["role"] == "tool"]
                self.assertEqual(shown[0], "Ok. The knob has been set.")
                self.assertRegex(shown[1], r"^0\.3\d{3} s$")      # just the time
                self.assertEqual(shown[4], "The knob cannot go above 2.")
                self.assertEqual(shown[7], "Plot 1 saved.")
                for name in ("transcript.jsonl", "measurements.csv", "notebook.md",
                             "summary.json", "plots/plot_01.png"):
                    self.assertTrue((self.folder / name).exists(), name)
                self.assertEqual((self.folder / "measurements.csv").read_text().splitlines()[0],
                                 "call,time,h,t")

    def test_a_wrong_law_is_not_found(self):
        script = [("tools", [("finish", {"conclusion": "linear", "formula": "0.3*h + 0.2",
                                         "constant": 0.3})])]
        summary, _ = self.session("openai", script)
        self.assertEqual(summary["status"], "finished")
        self.assertFalse(summary["discovered"])

    def test_text_tool_calls_and_the_call_limit(self):
        script = [("text", '{"name": "push_the_button", "arguments": {}}')] * 6
        summary, server = self.session("openai", script, max_calls=4)
        self.assertEqual((summary["status"], summary["drops"]), ("call_limit", 4))
        self.assertEqual(summary["text_tool_calls"], 4)

    def test_the_lab_period_warning_and_the_context_budget(self):
        script = [("tools", [("set_knob", {"h": 1.0}), ("push_the_button", {})])] * 40
        summary, server = self.session("openai", script, max_calls=40, warn_calls_left=5,
                                       context_tokens=1200)
        self.assertEqual(summary["lab_period_warning_call"], 36)
        self.assertGreater(summary["context"]["trimmed_calls"], 0)
        last = server.requests[-1]["messages"]
        self.assertEqual([m["role"] for m in last[:3]], ["system", "user", "assistant"])
        self.assertLess(len(last), 2 + 3 * 40)

    def test_a_model_that_never_uses_tools_is_stopped(self):
        summary, _ = self.session("openai", [("text", "Hmm.")] * 10, max_idle=3)
        self.assertEqual((summary["status"], summary["model_calls"]), ("no_tool_calls", 3))


class CommandLineTests(unittest.TestCase):
    def test_runs_folders_console_log_and_prompt(self):
        import run_lab
        server = FakeModelServer("openai", EXPERIMENT + EXPERIMENT)
        self.addCleanup(server.close)
        printed = io.StringIO()
        with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(printed):
            rc = run_lab.main(["--provider", "nrp", "--llm-url", server.url, "--model",
                               "fake:model", "--runs", "2", "--out-dir", out])
            model_dir = Path(out) / "fake_model"
            totals = json.loads((model_dir / "runs_summary.json").read_text())
            run1 = json.loads((model_dir / "001" / "summary.json").read_text())
            log1 = (model_dir / "001" / "console.log").read_text()
        self.assertEqual(rc, 0)
        self.assertEqual((totals["runs"], totals["discovered"]), (2, 2))
        self.assertIn("You have 2 controls.", run1["prompts"]["task"])
        self.assertIn("You'll hear a loud \"crash\"", run1["prompts"]["task"])
        self.assertEqual(run1["settings"]["context_tokens"], 8000)
        self.assertTrue(log1.lstrip().startswith("[run 1/2]"))
        self.assertIn("[run 001] finished, found the law", log1)
        first = server.requests[0]
        self.assertEqual(first["messages"][1]["content"],
                         (HERE / "prompts" / "task.txt").read_text().strip())


if __name__ == "__main__":
    unittest.main()
