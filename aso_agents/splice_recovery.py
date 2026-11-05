"""Retry a failed splice-model invocation in a fresh CUDA process."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def run_in_fresh_process(arguments, original_error):
    directory = Path(tempfile.mkdtemp(prefix="splice-recovery-",
                                     dir=Path(arguments["tiles_json_path"]).resolve().parent))
    request = directory / "request.json"
    output = directory / "result.json"
    log = directory / "worker.log"
    request.write_text(json.dumps(dict(arguments), ensure_ascii=False, indent=2))
    receipt = {"status": "started", "original_error": original_error, "attempts": 1,
               "input_sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
               "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "request": str(request), "output": str(output), "log": str(log)}
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print("Splice model: restarting scoring in a fresh process after a GPU error.", flush=True)
    environment = {**os.environ, "MODAL_ASO_SPLICE_WORKER": "1"}
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH"))))
    try:
        with log.open("w") as handle:
            completed = subprocess.run(
                [sys.executable, "-m", "aso_agents.splice_recovery", str(request), str(output)],
                env=environment,
                stdout=handle, stderr=subprocess.STDOUT, check=False)
    except OSError as exc:
        receipt.update(status="failed", error=str(exc))
        receipt_path.write_text(json.dumps(receipt, indent=2))
        raise RuntimeError(f"Splice-model GPU recovery could not start: {exc}. Details: {receipt_path}") from exc
    try:
        result = json.loads(output.read_text()) if output.is_file() else {}
    except (ValueError, OSError):
        result = {}
    expected_count = len(json.loads(Path(arguments["tiles_json_path"]).read_text())["rows"])
    valid = (isinstance(result, dict) and isinstance(result.get("results"), list)
             and len(result["results"]) == expected_count)
    success = completed.returncode == 0 and valid and not result.get("error")
    receipt.update(status="recovered" if success else "failed", returncode=completed.returncode)
    receipt_path.write_text(json.dumps(receipt, indent=2))
    if not success:
        error = result.get("error") if isinstance(result, dict) else None
        raise RuntimeError(f"Splice-model GPU recovery failed: {error or 'worker returned no valid result'}. Details: {log}")
    result["runtime_recovery"] = {**receipt, "receipt": str(receipt_path)}
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # Also guard direct worker invocations against recursive recovery.
    os.environ["MODAL_ASO_SPLICE_WORKER"] = "1"
    try:
        module = importlib.import_module("aso_agents.splicing_agent")
        result = module.evaluate_splice_shifts_on_tiles(**json.loads(args.request.read_text()))
        args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        args.output.write_text(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}))
        raise
    if result.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
