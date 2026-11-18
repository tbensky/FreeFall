# FreeFall: can an LLM discover the law of a falling ball?

A ball is held at a height h above a table (a knob, 0.1 to 2 m). Pushing a
button drops it and times the fall. The times follow

    h = (1/2) g t^2,   t = sqrt(2 h / g),   g = 9.8 m/s^2

plus Gaussian timing noise. The model is told none of that. It gets this
task (`prompts/task.txt`, word for word from `design.txt`):

> You have 2 controls. One is a knob that goes from 0.1 to 2. These numbers are
> in meters. The other is a push button. Depending on your knob setting, when
> you push the button, you'll see a ball fall from the height of the knob
> setting. You'll hear a loud "crash" when the ball hits the table, and get a
> single time, in seconds. This is your signal. There is a law of physics in
> use here, and a constant that I want you to discover. ... Also, one
> requirement is that the constant is to be determine by some kind of curve
> fit of data you decide take. Do not determine the constant by doing a lot of
> averaging or mathematical manipulation of individual settings or data.

It is built like Malus (`../Malus`): the same model clients, notebook, plot
tool, context window and run folders.

## Quick start

Standard library only (Python 3.10+); matplotlib, if installed, makes the
plots PNGs.

```bash
# a hosted model on NRP, with the same LLM_* settings as GASolver and Malus
export LLM_API_KEY=...                       # your NRP key
python3 run_lab.py --provider nrp --model deepseek-v4-flash

# a local Ollama model, 10 independent runs
python3 run_lab.py --provider ollama --model qwen3:8b --runs 10
```

`--provider` is `nrp`, `openai` or `deepseek` (any OpenAI-compatible
`/v1/chat/completions` endpoint; `--llm-url` for others) or `ollama`
(`localhost:11434`). `LLM_PROVIDER`, `LLM_URL`, `LLM_MODEL` and `LLM_API_KEY`
set the defaults. A host error is retried up to 8 times (2, 4, 8, 16, then
30 s waits).

## The tools

| Tool | What the model reads, and what it gets back |
| --- | --- |
| `set_knob(h)` | Turn the knob to a setting from 0.1 to 2. Replies only "Ok. The knob has been set." (or why not, e.g. "The knob cannot go above 2."). |
| `push_the_button()` | Drops the ball. Replies only the time, e.g. `0.4518 s`; keeping track of the setting is the model's job. |
| `write_down(text)` | Adds a notebook entry; the text is kept verbatim and no action is taken on it. |
| `read_notebook()` | Every notebook entry. |
| `plot(points, formula?, labels?)` | Graphs `[x, y]` points the model types itself, with an optional curve in `x`, like a spreadsheet. Replies only "Plot N saved.": no curve values, differences or RMS, so judging a fit is the model's own work. |
| `finish(conclusion, formula, constant)` | Ends the run: its findings, its law as a Python expression for the time in terms of h, and the value of its constant. |

`plot` is not in the original list; it is there because the task asks for the
constant from a curve fit, and it gives that fit a record. Unlike Malus there
is no fit gate or data check: `finish` always ends the run.

## The model's context

Each call sends the system prompt, the task and as many of the latest
exchanges as fit in **8000 tokens** (`--context-tokens`, tool definitions
included). Older exchanges are dropped without a word, so older results live
on only in the notebook. Tokens are estimated from characters and calibrated
from the host's own counts; the latest exchange is always sent. `0` sends the
whole conversation. For Ollama, `--num-ctx` (16384) must leave room for that
plus the reply.

A run gets 200 model calls (`--max-calls`). With 20 left
(`--warn-calls-left`), the model is told once: "The lab period is almost
over: you have 20 turns left. Finish your curve fit, record your constant in
your notebook, and call finish." (`prompts/lab_period.txt`).

## Scoring

The law in `finish` is parsed with a whitelist (numbers, `h`, arithmetic,
`math` functions) and compared with 30 fresh timings, each the mean of 3
drops, at random heights the model never chose. The RMS error divided by the
range of the timings is `nrmse`; 0.05 or less counts as found
(`--threshold`).

The constant a model reports could be g, g/2, 2/g or sqrt(2/g), all correct,
so it is recorded as given (`constant`) but not compared. Instead the summary
gives `implied_g`, the g its law implies, 2h/t(h)^2, at the check heights. A
law of the right form gives the same value at every height, and it should be
close to 9.8.

## What a run produces

```text
Runs/<model>/001/transcript.jsonl   every message, tool call and result
                 console.log        what was printed on the terminal
                 measurements.csv   every drop: call, time, h, t
                 notebook.md        what it wrote down (also notebook.jsonl)
                 plots/plot_01.png  each graph it made (plots.jsonl has the data)
                 summary.json       outcome, law, constant, implied g, settings,
                                    the exact prompts and tool definitions
Runs/<model>/runs_summary.json      how many runs found the law
```

Statuses: `finished`, `call_limit`, `no_tool_calls` (five replies in a row
without a tool), `error` (the endpoint kept failing) and `interrupted`.

## The simulated rig

| Option | Default | |
| --- | --- | --- |
| `--g` | 9.8 | m/s^2 |
| `--noise` | 0.002 | timing noise, seconds (falls take 0.14 to 0.64 s) |
| `--timer-offset` | 0 | a constant added to every timing (release latency), seconds |
| `--seed` | 1 | run NNN uses seed + NNN − 1, so runs repeat exactly |

The knob starts at 0.1 until the model turns it.

## Connecting a real rig

Everything that knows about the physics is in `freefall/apparatus.py`. The
tools, the loop and the check use only the `Apparatus` interface:

```python
class Apparatus:
    min_height, max_height        # 0.1 and 2.0 m
    def set_height(h)             # move the release point, block until there
    def get_height()              # where it is now
    def drop()                    # release the ball, return the timer's reading (s)
    def close()                   # park the rig
```

`HardwareApparatus` is a stub with those methods; fill them in for the height
stage, the release and the timer, then run with `--apparatus hardware`. Raise
`ApparatusError` with a plain message for anything the rig cannot do; the
model sees it as the tool's reply. The final check takes its fresh timings
through the same interface.

## Layout

```text
run_lab.py              command line: runs, folders, console.log, summaries
freefall/apparatus.py   Apparatus interface, simulated rig, hardware stub
freefall/tools.py       the six tools
freefall/lab.py         the conversation loop, context window, run folders
freefall/evaluate.py    safe formula parsing, the check, implied g
freefall/llm.py         Ollama and OpenAI-compatible tool calling (from Malus)
freefall/notebook.py    the lab notebook
freefall/plotting.py    drawing a plot
prompts/                the task, the system prompt, the lab-period warning
tests/test_lab.py       offline tests against fake OpenAI and Ollama servers
```

Run the tests with `python3 -m unittest tests.test_lab`.
