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
import concurrent.futures
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
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

# --- R2: use the allowance the rules grant -----------------------------------------------
# docs/PREREG-R2.md. The route admits 25 generation requests per unit and permits concurrent
# requests (track1-coding-public#15); organizer records for a team that completed all 86 units
# show a median of 24 requests per unit against our 4-8. The ceiling here is deliberately 8,
# not 25: a request is charged at admission and never refunded, so headroom is kept for
# upstream failures rather than spent.
N_CANDIDATES = int(os.environ.get("AGENT_N_CANDIDATES", "5"))
N_REPAIRS = int(os.environ.get("AGENT_N_REPAIRS", "2"))
CAND_RUN_SEC = float(os.environ.get("AGENT_CAND_RUN_SEC", "60"))
R2 = os.environ.get("AGENT_R2", "1") != "0"      # AGENT_R2=0 restores the R1 serial loop

# The per-unit ceiling PREREG-R2 commits to. It has to bind the WHOLE unit, serial fallback
# included: the first sweep spent 6 on candidates + 2 on repairs and then another 4 in the
# fallback, i.e. 12, because the fallback had no idea what R2 had already spent. Lowering the
# shared allowance is what makes the stated ceiling real rather than aspirational.
if R2 and "AGENT_MAX_REQUESTS" not in os.environ:
    MAX_REQUESTS = N_CANDIDATES + 1 + N_REPAIRS                  # 5 + checker + 2 = 8

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
        # R2 sends several requests at once, so the allowance check has to be atomic: the
        # route itself counts exactly under concurrency ("with one slot left, two simultaneous
        # requests get one admitted and one refused"), and this mirrors that locally so we
        # never spend a slot we did not mean to.
        self._lock = threading.Lock()
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
        with self._lock:
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


def run_script(code: str, out: pathlib.Path, work: pathlib.Path,
               budget: float | None = None, name: str = "solution.py") -> tuple[int, str]:
    # `budget` lets R2 give each of several candidates a smaller slice than one serial attempt
    # would take; `name` keeps the checker script off the solution path. Both default to the
    # R1 behaviour, so the serial fallback is unchanged.
    script = work / name
    script.write_text(code)
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(out),                       # relative writes land in the output tree
            capture_output=True, text=True,
            timeout=(min(RUN_TIMEOUT_SEC, max(20.0, left() - 20.0))
                     if budget is None else max(5.0, budget)),
        )
        tail = (proc.stdout[-1500:] + "\n" + proc.stderr[-3000:]).strip()
        return proc.returncode, tail
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT: the script exceeded its execution window"
    except Exception:
        return 125, traceback.format_exc()[-3000:]


# --- R2: candidates, an independent checker, and agreement -------------------------------
# docs/PREREG-R2.md. Two proxies stand in for the real checker, which is sealed:
#   A. a checker script the model writes from instruction.md ALONE -- it never sees a
#      candidate solution, so it is not marking its own work;
#   B. agreement between independently generated solutions -- these tasks are deterministic
#      numerics, so wrong implementations disagree in different ways and right ones do not.
# Diversity comes from distinct prompt variants, NOT from sampling: temperature stays 0 so two
# offline runs of this agent stay byte-identical, which is what the starter pack asks for.

VARIANTS = (
    "",
    "Prefer a closed-form or analytic formulation wherever the task admits one.",
    "Prefer an explicit numerical method (grid, finite difference or simulation) and keep the "
    "discretisation fine enough that the stated tolerances hold.",
    "Be defensive about the data first: validate the input schema, and handle the edge cases "
    "the task names (very short maturities, deep in/out-of-the-money, missing or zero values) "
    "explicitly rather than letting them propagate.",
    "Before writing the final values, re-derive the units, sign conventions and per-day/per-year "
    "scaling the task specifies, and apply them exactly as written.",
)

ASSERT_SYSTEM = (
    "You write pytest-free Python 3 validation scripts. Given a task description, you emit one "
    "self-contained script that READS the produced output files and ASSERTS the properties the "
    "task requires: the named files exist, the named columns are present, row counts match the "
    "input, dtypes are right, and every stated range, sign and relationship holds. The script "
    "computes nothing of its own beyond what an assertion needs, never writes any file, and "
    "exits non-zero on the first violated assertion. Return ONLY the script in one ```python "
    "code block."
)


def build_assert_prompt(instruction: str, files: str, out_dir: str) -> str:
    return (
        f"{instruction}\n\n"
        f"## Files present under /input\n{files}\n\n"
        f"## What to validate\n"
        f"- The deliverables have been written to `{out_dir}` (also reachable as `/output`).\n"
        f"- Assert every property the task states about them. Do not recompute the answer; "
        f"assert only what the task itself guarantees.\n"
        f"- Write nothing. Print a one-line summary, then exit 0 if all assertions hold.\n"
    )


def gen_parallel(model: "Model", system: str | list[str], prompts: list[str]) -> list[str | None]:
    """Issue several generation requests at once. The route permits this and counts exactly."""
    out: list[str | None] = [None] * len(prompts)
    if not prompts:
        return out

    def one(i: int) -> None:
        try:
            request_system = system[i] if isinstance(system, list) else system
            out[i] = model.chat([{"role": "system", "content": request_system},
                                 {"role": "user", "content": prompts[i]}])
        except Exception as exc:                                  # never propagate
            log(f"candidate {i} raised: {exc!r}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        list(ex.map(one, range(len(prompts))))
    return out


def stash_run(code: str, out: pathlib.Path, work: pathlib.Path, tag: str,
              budget: float) -> dict:
    """Run one candidate, then move whatever it produced out of the output tree.

    Generated code writes to /output/... by absolute path because the instruction says so, so
    candidates cannot be isolated by running them elsewhere; they are run one at a time and
    their products are swept into a stash between runs. The output tree is left exactly as it
    was found, so the winner can be restored into a clean directory at the end.
    """
    before = deliverables(out)
    rc, tail = run_script(code, out, work, budget)
    made = sorted(deliverables(out) - before)
    stash = work / f"stash-{tag}"
    shutil.rmtree(stash, ignore_errors=True)
    stash.mkdir(parents=True, exist_ok=True)
    for rel in made:
        src, dst = out / rel, stash / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except Exception as exc:
            log(f"{tag}: could not stash {rel}: {exc!r}")
    return {"rc": rc, "tail": tail, "files": made, "stash": stash}


def restore(stash: pathlib.Path, out: pathlib.Path) -> None:
    for src in sorted(stash.rglob("*")):
        if src.is_file():
            dst = out / src.relative_to(stash)
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
            except Exception as exc:
                log(f"could not restore {src.name}: {exc!r}")


def _table(path: pathlib.Path):
    try:
        import pandas as pd
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in (".csv", ".tsv"):
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def agrees(a: dict, b: dict) -> bool:
    """True when two candidates produced numerically equivalent tables."""
    if not a["files"] or a["files"] != b["files"]:
        return False
    try:
        import numpy as np
        for rel in a["files"]:
            ta, tb = _table(a["stash"] / rel), _table(b["stash"] / rel)
            if ta is None or tb is None:
                if (a["stash"] / rel).read_bytes() != (b["stash"] / rel).read_bytes():
                    return False
                continue
            if ta.shape != tb.shape:
                return False
            # pandas extension dtypes (StringDtype and friends) make np.issubdtype raise,
            # so ask pandas, not numpy, what is numeric.
            from pandas.api.types import is_numeric_dtype
            num = [c for c in ta.columns
                   if c in tb.columns and is_numeric_dtype(ta[c]) and is_numeric_dtype(tb[c])]
            if not num:
                return False
            for c in num:
                if not np.allclose(ta[c].to_numpy(dtype=float),
                                   tb[c].to_numpy(dtype=float), rtol=1e-3, atol=1e-8,
                                   equal_nan=True):
                    return False
        return True
    except Exception as exc:
        log(f"agreement check failed: {exc!r}")
        return False


def check_passes(checker: str | None, cand: dict, out: pathlib.Path,
                 work: pathlib.Path) -> bool:
    """Restore a candidate, run the model-written checker against it, sweep it back out."""
    if not checker or not cand["files"]:
        return False
    restore(cand["stash"], out)
    try:
        before = deliverables(out)
        rc, tail = run_script(checker, out, work, min(45.0, max(10.0, left() - 30.0)),
                              name="checker.py")
        # a checker must not create anything; if it did, remove it
        for rel in sorted(deliverables(out) - before):
            try:
                (out / rel).unlink()
            except OSError:
                pass
        if rc != 0:
            log(f"checker rejected: {tail.strip().splitlines()[-1][:160] if tail.strip() else rc}")
        return rc == 0
    finally:
        for rel in cand["files"]:
            try:
                (out / rel).unlink()
            except OSError:
                pass


def solve_r2(task_dir: pathlib.Path, out: pathlib.Path, work: pathlib.Path,
             model: "Model", instruction: str, files: str) -> bool:
    """Returns True when a candidate was accepted and its files are in `out`."""
    base = build_prompt(instruction, files, str(out))
    n = max(1, min(N_CANDIDATES, MAX_REQUESTS - 1))
    prompts = [base + (f"\n## Approach\n{v}\n" if v else "") for v in VARIANTS[:n]]

    log(f"R2: {n} candidates + 1 checker, in parallel ({left():.0f}s left)")
    # The checker only reads the instruction, so its generation need not wait for candidates.
    generated = gen_parallel(model, [SYSTEM] * len(prompts) + [ASSERT_SYSTEM],
                             prompts + [build_assert_prompt(instruction, files, str(out))])
    replies, checker_reply = generated[:-1], generated[-1]
    checker = extract_code(checker_reply) if checker_reply else None
    log(f"R2: {sum(r is not None for r in replies)}/{n} candidates returned, "
        f"checker {'ok' if checker else 'missing'}, {model.calls} requests used")

    cands: list[dict] = []
    for i, reply in enumerate(replies):
        if left() < 90:
            log(f"R2: stopping candidate runs, {left():.0f}s left")
            break
        code = extract_code(reply) if reply else None
        if code is None:
            continue
        c = stash_run(code, out, work, f"c{i}", min(CAND_RUN_SEC, max(20.0, left() - 60.0)))
        c["code"] = code
        c["ok"] = c["rc"] == 0 and bool(c["files"])
        log(f"R2: candidate {i} exit={c['rc']} files={c['files'] or 'none'}")
        cands.append(c)
        if c["ok"] and check_passes(checker, c, out, work):
            log(f"R2: candidate {i} passed the checker; shipping it")
            restore(c["stash"], out)
            return True

    # No candidate cleared the checker. Repair the most promising ones, in parallel.
    broken = [c for c in cands if not c["ok"]][:N_REPAIRS]
    if broken and left() > 120 and model.calls + len(broken) <= MAX_REQUESTS:
        log(f"R2: repairing {len(broken)} candidate(s) ({left():.0f}s left)")
        rp = [base + (
            f"\n## A previous attempt failed\n```python\n{c['code'][:6000]}\n```\n"
            f"It exited {c['rc']} and produced {c['files'] or 'no files'}.\n"
            f"Captured output (tail):\n```\n{c['tail'][:2500]}\n```\n"
            f"Return ONLY the corrected full script in one ```python block.\n") for c in broken]
        for j, reply in enumerate(gen_parallel(model, SYSTEM, rp)):
            if left() < 90:
                break
            code = extract_code(reply) if reply else None
            if code is None:
                continue
            c = stash_run(code, out, work, f"r{j}", min(CAND_RUN_SEC, max(20.0, left() - 40.0)))
            c["code"], c["ok"] = code, c["rc"] == 0 and bool(c["files"])
            log(f"R2: repair {j} exit={c['rc']} files={c['files'] or 'none'}")
            cands.append(c)
            if c["ok"] and check_passes(checker, c, out, work):
                log(f"R2: repair {j} passed the checker; shipping it")
                restore(c["stash"], out)
                return True

    # Still nobody cleared the checker: fall back to agreement, then to any clean run.
    good = [c for c in cands if c["ok"]]
    if not good:
        log("R2: no candidate produced output")
        return False
    if len(good) > 1:
        votes = [sum(1 for o in good if o is not c and agrees(c, o)) for c in good]
        best = max(range(len(good)), key=lambda i: votes[i])
        log(f"R2: no checker pass; agreement votes={votes}, shipping candidate with {votes[best]}")
        restore(good[best]["stash"], out)
        return True
    log("R2: one clean candidate, shipping it")
    restore(good[0]["stash"], out)
    return True


def solve(task_dir: pathlib.Path, out: pathlib.Path) -> None:
    work = pathlib.Path(os.environ.get("AGENT_WORK_DIR", "/tmp/agent-work"))
    work.mkdir(parents=True, exist_ok=True)

    instruction = (task_dir / "instruction.md").read_text(errors="replace")
    files = survey(task_dir)
    model = Model()
    if not model.available():
        log("NO MODEL ACCESS (MODEL_ENDPOINT unset and no stub) — nothing can be earned here")
        return

    if R2:
        try:
            if solve_r2(task_dir, out, work, model, instruction, files):
                return
            log("R2 produced nothing; falling through to the serial loop")
        except Exception:
            # A fault in the new path must cost at most the requests it already spent.
            log("R2 raised; falling back to the serial loop\n" + traceback.format_exc()[-1200:])
        for rel in sorted(deliverables(out)):          # start the fallback from a clean tree
            try:
                (out / rel).unlink()
            except OSError:
                pass

    solve_serial(out, model, instruction, files, work)


def solve_serial(out: pathlib.Path, model: "Model", instruction: str, files: str,
                 work: pathlib.Path) -> None:
    """The R1 loop, unchanged: one script at a time, traceback fed back, up to MAX_ROUNDS."""
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
