"""Track 1 coding agent — minimal but real.

Contract (SUBMISSION_CLI / baselines/README):
  solve --task-dir /input --out /app/output
`/input` is the whole unit directory (instruction.md, environment/data/..., card.toml).
`/app/output` and `/output` are two binds of ONE host directory.

Three constraints drive every design choice here, and two of them push the OPPOSITE way
from the kit's "where the headroom is" list:

1. **The roster clock, not the unit clock, is binding.** A run gets ONE 12-hour wall-clock
   allowance for all 86 units (~8 min/unit), while a single card allows 1200-5400s. A greedy
   multi-turn loop that spends 25 requests and 30 minutes on one unit starves the units behind
   it, and every unit the run never reaches is scored 0 (`not_reached`). So this agent carries
   its own budget and always leaves time to write something.
2. **Rule 9**: a pass only counts if the House model did the solving at run time. No stored
   answers, no heuristic fallback that bypasses the model.
3. **25 admitted requests per unit, <=4000 output tokens each.** Retries and failures count.

Everything is wrapped: the agent never raises and always exits 0, because a non-zero exit is
scored `container_crashed` whatever the output tree holds.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback

# --- budgets -----------------------------------------------------------------------------
# Default 420s: ~8 min/unit is the roster-clock share, and the checker still needs its own
# window afterwards. Override per experiment with AGENT_BUDGET_SEC.
BUDGET_SEC = float(os.environ.get("AGENT_BUDGET_SEC", "420"))
MAX_ROUNDS = int(os.environ.get("AGENT_MAX_ROUNDS", "4"))
MAX_REQUESTS = int(os.environ.get("AGENT_MAX_REQUESTS", "20"))   # hard cap is 25; stay under
MAX_OUTPUT_TOKENS = 4000                                         # rule cap, do not raise
RUN_TIMEOUT_SEC = float(os.environ.get("AGENT_RUN_TIMEOUT_SEC", "120"))

T0 = time.time()


def left() -> float:
    return BUDGET_SEC - (time.time() - T0)


def log(msg: str) -> None:
    print(f"[agent {time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


# --- input survey ------------------------------------------------------------------------

def survey(task_dir: pathlib.Path, limit: int = 40) -> str:
    """A short, factual description of what is under /input.

    The model cannot list the filesystem, and most failures in a task like this are schema
    failures, not algorithm failures. `checks/` is deliberately skipped: it is sealed in the
    real evaluation, and designing around it would not transfer.
    """
    lines: list[str] = []
    for p in sorted(task_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(task_dir).as_posix()
        if rel.startswith("checks/") or rel in ("instruction.md",):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        lines.append(f"- /input/{rel}  ({size} bytes)")
        if p.suffix in (".parquet", ".csv") and len(lines) < limit:
            lines.append(_peek(p))
        if len(lines) >= limit:
            lines.append("- ... (truncated)")
            break
    return "\n".join(lines) if lines else "(no files besides instruction.md)"


def _peek(p: pathlib.Path) -> str:
    try:
        import pandas as pd
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, nrows=200)
        cols = ", ".join(f"{c}:{df[c].dtype}" for c in df.columns[:30])
        return f"    shape={df.shape}, columns=[{cols}]"
    except Exception as exc:                                    # never fatal
        return f"    (could not read: {exc!r})"


SYSTEM = (
    "You are a quantitative-finance engineer. You write one self-contained Python 3 script "
    "that reads its inputs from /input and writes the required deliverables to the output "
    "directory. The script runs offline: numpy, pandas, scipy, pyarrow and numba are available, "
    "nothing can be installed, and there is no network. Return ONLY the script inside one "
    "```python code block, with no prose before or after. The script must be deterministic, "
    "must handle the stated numerical edge cases without raising, and must write every file the "
    "task names, with exactly the named columns, dtypes and row order."
)


def build_prompt(instruction: str, files: str, out_dir: str) -> str:
    return (
        f"{instruction}\n\n"
        f"## Files present under /input\n{files}\n\n"
        f"## Execution contract\n"
        f"- Write deliverables to `{out_dir}` (also reachable as `/output`; same directory).\n"
        f"- The script is run with that directory as the working directory, so a relative path "
        f"lands there too. Prefer absolute paths.\n"
        f"- Do NOT write reward.json or pytest_report.json; the checker writes those.\n"
        f"- Finish well under {int(RUN_TIMEOUT_SEC)} seconds of CPU time.\n"
    )


CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    blocks = CODE_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    # A bare script with no fence is still usable if it looks like Python.
    if text and ("import " in text or "def " in text):
        return text.strip()
    return None


# --- model ------------------------------------------------------------------------------

class Model:
    """House endpoint, or a local stand-in for plumbing tests.

    The House route exists only inside the scored harness, so STUB mode is what makes the rest
    of this agent testable at all: AGENT_STUB_FILE supplies a canned reply and every other code
    path runs unchanged. It is a development aid only — a stubbed run earns nothing (rule 9).
    """

    def __init__(self) -> None:
        self.calls = 0
        self.stub = os.environ.get("AGENT_STUB_FILE")
        self.endpoint = os.environ.get("MODEL_ENDPOINT")
        self.client = None
        if not self.stub and self.endpoint:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.endpoint.rstrip("/") + "/v1",
                api_key=os.environ.get("MODEL_TOKEN", ""),
                timeout=float(os.environ.get("AGENT_HTTP_TIMEOUT", "180")),
                max_retries=0,          # retries count against the 25-request budget
            )
            self.name = os.environ.get("MODEL_NAME", "house")

    def available(self) -> bool:
        return bool(self.stub or self.client)

    def chat(self, messages: list[dict]) -> str | None:
        if self.calls >= MAX_REQUESTS:
            log(f"request cap reached ({self.calls})")
            return None
        self.calls += 1
        if self.stub:
            return pathlib.Path(self.stub).read_text()
        try:
            r = self.client.chat.completions.create(
                model=self.name,
                messages=messages,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0,
            )
            return r.choices[0].message.content
        except Exception as exc:
            log(f"model call {self.calls} failed: {exc!r}")
            return None


# --- the loop ---------------------------------------------------------------------------

def deliverables(out: pathlib.Path) -> set[str]:
    skip = {"reward.json", "pytest_report.json", "solution.py"}
    return {
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and p.name not in skip
    }


def run_script(code: str, out: pathlib.Path, work: pathlib.Path) -> tuple[int, str]:
    script = work / "solution.py"
    script.write_text(code)
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(out),                       # relative writes land in the output tree
            capture_output=True, text=True,
            timeout=min(RUN_TIMEOUT_SEC, max(20.0, left() - 20.0)),
        )
        tail = (proc.stdout[-1500:] + "\n" + proc.stderr[-3000:]).strip()
        return proc.returncode, tail
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT: the script exceeded its execution window"
    except Exception:
        return 125, traceback.format_exc()[-3000:]


def solve(task_dir: pathlib.Path, out: pathlib.Path) -> None:
    work = pathlib.Path(os.environ.get("AGENT_WORK_DIR", "/tmp/agent-work"))
    work.mkdir(parents=True, exist_ok=True)

    instruction = (task_dir / "instruction.md").read_text(errors="replace")
    files = survey(task_dir)
    model = Model()
    if not model.available():
        log("NO MODEL ACCESS (MODEL_ENDPOINT unset and no stub) — nothing can be earned here")
        return

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_prompt(instruction, files, str(out))},
    ]

    before = deliverables(out)
    for rnd in range(1, MAX_ROUNDS + 1):
        if left() < 60:
            log(f"stopping before round {rnd}: {left():.0f}s left")
            break
        log(f"round {rnd}: asking the model ({left():.0f}s left)")
        reply = model.chat(messages)
        if reply is None:
            break
        code = extract_code(reply)
        if code is None:
            log("no code block in reply")
            messages += [
                {"role": "assistant", "content": reply[:2000]},
                {"role": "user", "content": "Return ONLY one ```python code block."},
            ]
            continue

        rc, tail = run_script(code, out, work)
        made = deliverables(out) - before
        log(f"round {rnd}: exit={rc}, new files={sorted(made) or 'none'}")
        if rc == 0 and made:
            log("deliverables present; done")
            return

        problem = (
            f"The script exited {rc} and produced "
            f"{'no new files' if not made else 'files ' + str(sorted(made))}.\n"
            f"Captured output (tail):\n```\n{tail}\n```\n"
            f"Fix the script and return ONLY the corrected full script in one ```python block."
        )
        messages += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                     {"role": "user", "content": problem}]

    log(f"finished without a clean run; files now: {sorted(deliverables(out))}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="solve")
    p.add_argument("verb", nargs="?", default="solve")
    p.add_argument("--task-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.verb != "solve":
        print(f"unknown verb {args.verb!r}", file=sys.stderr)
        return 2
    out = pathlib.Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
        solve(pathlib.Path(args.task_dir), out)
    except Exception:
        # A non-zero exit is scored container_crashed whatever the tree holds, so never raise.
        traceback.print_exc(file=sys.stderr)
    log(f"exit 0 after {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
