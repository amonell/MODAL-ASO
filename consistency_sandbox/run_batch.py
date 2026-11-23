"""Capture fresh MODAL-ASO runs using the local WSL runtime."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone


SANDBOX = Path(__file__).resolve().parent
REPO = SANDBOX.parent
CASES = ("c3", "mef2c", "angpt2")
RUN_FLAGS = ("--mode", "agentic", "--agentic-max-steps", "120",
             "--router-max-steps", "25", "--router-log-tool-calls", "--raw-cli", "--no-stream-ui")
RECORDED_SETTINGS = (
    "ASO_LLM_PROVIDER", "OPENROUTER_MODEL", "OPENROUTER_REASONING_EFFORT",
    "OPENROUTER_ONLY_PROVIDERS", "OPENROUTER_PROVIDER_ORDER", "OPENROUTER_PROVIDER_SORT",
    "OPENROUTER_ZDR", "OPENROUTER_DATA_COLLECTION", "ASO_LLM_TRANSPORT_RETRIES",
    "ASO_GOAL_INTAKE_MODEL", "ASO_REQUEST_WORKBENCH_MODEL",
    "ASO_WORKBENCH_RESPONSE_MODEL", "ASO_SANDBOX_ANALYSIS_MODEL",
    "ASO_ANSWER_WEB_SEARCH_MODEL",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def read_prompt(path):
    return path.read_bytes().decode("utf-8")


def worker(run_dir, case, data_dir):
    sys.path.insert(0, str(REPO))
    from aso_agents.cli import _run

    reply_used = False

    def clarify(*, questions, context):
        nonlocal reply_used
        question = questions[0] if len(questions) == 1 else ""
        replay = (
            case == "mef2c" and not reply_used
            and re.search(r"\b(tissue|cell type|disease model|broad baseline|biological context)\b",
                          question, flags=re.IGNORECASE)
            and not re.search(r"\b(transcript|genome build|assembly|coordinate|chemistry)\b",
                              question, flags=re.IGNORECASE)
        )
        answer = read_prompt(run_dir / "reply.txt") if replay else None
        reply_used = reply_used or bool(replay)
        with (run_dir / "clarifications.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"questions": questions, "context": context, "reply": answer},
                                 default=str) + "\n")
        return [{"question": question, "answer": answer}] if answer is not None else None

    args = ["--output-dir", str(run_dir / "output"), *RUN_FLAGS]
    if data_dir:
        args.extend(["--data-dir", str(data_dir)])
    return _run(
        [*args, "--", read_prompt(run_dir / "prompt.txt")],
        clarification_resolver=clarify,
        result_callback=lambda result: write_json(run_dir / "result.json", result),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append", help="Run only the specified cases")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--data-dir", type=Path, help="Existing MODAL-ASO runtime data directory")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without starting MODAL-ASO")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = tuple(dict.fromkeys(args.case or CASES))
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    data_dir = args.data_dir.resolve() if args.data_dir else None
    if args.worker:
        return worker(args.worker, cases[0], data_dir)

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    batch = SANDBOX / "captures" / batch_id
    # Rotate the starting prompt to spread early and late positions across cases.
    plan = [(cases[(index + repeat - 1) % len(cases)], repeat)
            for repeat in range(1, args.repeats + 1) for index in range(len(cases))]
    if args.dry_run:
        for case, repeat in plan:
            print(batch / case / f"repeat_{repeat:02d}")
        return 0
    if sys.platform != "linux":
        parser.error("run this script using the existing MODAL-ASO Python environment inside WSL2")

    prompts = {case: read_prompt(SANDBOX / "prompts" / f"{case}.txt") for case in cases}
    reply = read_prompt(SANDBOX / "prompts" / "mef2c_reply.txt") if "mef2c" in cases else None
    env = os.environ.copy()
    env["MODAL_ASO_SKIP_RUNTIME_INSTALL"] = "1"
    batch.mkdir(parents=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    working_tree = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                  capture_output=True, text=True, check=True).stdout.splitlines()
    write_json(batch / "batch.json", {
        "batch_id": batch_id, "plan": plan, "git_revision": revision,
        "git_status": working_tree, "python": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "run_flags": RUN_FLAGS, "data_dir": data_dir,
        "model_environment": {key: env[key] for key in RECORDED_SETTINGS if key in env},
    })
    incomplete = 0
    for case, repeat in plan:
        run_dir = batch / case / f"repeat_{repeat:02d}"
        run_dir.mkdir(parents=True)
        scratch = run_dir / "tmp"
        scratch.mkdir()
        (run_dir / "prompt.txt").write_bytes(prompts[case].encode("utf-8"))
        if case == "mef2c":
            (run_dir / "reply.txt").write_bytes(reply.encode("utf-8"))
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(run_dir),
                   "--case", case]
        if data_dir:
            command.extend(["--data-dir", str(data_dir)])
        started = datetime.now(timezone.utc).isoformat()
        print(f"Running {case} repeat {repeat}: {run_dir}", flush=True)
        with (run_dir / "console.log").open("wb") as log:
            process = subprocess.run(command, cwd=REPO, env={**env, "TMPDIR": str(scratch)},
                                     stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
        result_path = run_dir / "result.json"
        status = json.loads(result_path.read_text(encoding="utf-8")).get("status") if result_path.exists() else None
        write_json(run_dir / "capture.json", {
            "case": case, "repeat": repeat, "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "exit_code": process.returncode, "status": status,
        })
        incomplete += process.returncode != 0 or status not in {"run_complete", "completed"}
        print(f"  {status or 'no result'} (exit {process.returncode})", flush=True)
    print(f"Captured {len(plan)} attempts; {incomplete} did not complete. {batch}")
    return int(incomplete > 0)


if __name__ == "__main__":
    sys.exit(main())
