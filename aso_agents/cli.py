"""Command-line setup, asset inspection and natural-language design requests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from . import __version__
from .paths import (data_root, reference_root, cache_root, oligoai_root,
                    resource_root, save_settings, splice_root)


def _configure(args, *, persist=False):
    if getattr(args, "data_dir", None):
        directory = str(Path(args.data_dir).expanduser().resolve())
        os.environ["MODAL_ASO_DATA_DIR"] = directory
        if persist:
            save_settings({"data_dir": directory})


def _setup(args):
    _configure(args, persist=True)
    features = set(args.features)
    if "all" in features:
        features = {"tools", "references", "splice", "oligoai"}
    if "oligoai" in features and (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("OligoAI requires Linux x86-64 with an NVIDIA Ampere or newer GPU. Use --features tools references splice on Apple silicon.")
    print(f"Data: {data_root()}")
    if "tools" in features:
        from .setup_tools import install_tools
        install_tools()
    from .runtime_binaries import refresh_runtime_path
    refresh_runtime_path()
    if "references" in features:
        from .reference_assets import install_reference
        for species in args.species:
            install_reference(species)
    if "splice" in features:
        if importlib.util.find_spec("gdown") is None:
            raise RuntimeError('Install the splice extra first: uv sync --extra splice (source) or pip install "modal-aso[splice]" (wheel).')
        if "human" in args.species:
            from .splice_eval_assets import ensure_splicetransformer_assets
            result = ensure_splicetransformer_assets()
            if result["status"] != "ok":
                raise RuntimeError(result["error"])
            print("Ready: SpliceTransformer")
        if "mouse" in args.species:
            from .openspliceai_assets import install_mouse_model
            install_mouse_model()
    if "oligoai" in features:
        from .oligoai_runtime import install
        install(oligoai_root(), Path(sys.executable))
        print("Ready: OligoAI")
    print("Setup complete. Run modal-aso doctor to check this environment.")
    return 0


def _doctor(args):
    _configure(args)
    from .runtime_binaries import refresh_runtime_path
    from .setup_tools import BINARIES
    from .reference_assets import inventory
    refresh_runtime_path()
    checks = []
    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": str(detail)})
    add("Python", "ok" if sys.version_info[:2] == (3, 11) else "error", platform.python_version())
    for name in BINARIES + ["snakemake"]:
        path = shutil.which(name)
        add(name, "ok" if path else "missing", path or "Run modal-aso setup --features tools")
    for name in ["tiling/Snakefile", "tiling/filter_seqs.py", "tiling/filter_mapped.py", "tiling/tile_seqs.py", "scripts/run_openspliceai_eval.py"]:
        add(name, "ok" if (resource_root() / name).is_file() else "error", resource_root() / name)
    for species in args.species:
        values = inventory(species, full=args.verify)
        missing = [v for v in values if v["status"] != "verified"]
        add(species + " references", "ok" if not missing else "missing", f"{len(values) - len(missing)}/{len(values)} verified; {reference_root()}")
    from .model_assets import inventory as model_inventory
    for item in model_inventory(args.species, full=args.verify):
        add(item["name"], "ok" if item["status"] == "verified" else item["status"], item["path"])
    import torch
    add("Inference device", "ok", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU; OligoAI requires a supported NVIDIA GPU")
    from .llm_provider import get_provider_config
    provider = get_provider_config(require_credentials=False)
    add("LLM credentials", "ok" if provider.api_key else "missing", provider.name + " (key configured)" if provider.api_key else provider.name + " (set API key)")
    result = {"version": __version__, "data": str(data_root()), "cache": str(cache_root()), "checks": checks}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"MODAL-ASO {__version__}\n")
        for check in checks:
            print(f"{check['status'].upper():8} {check['name']}: {check['detail']}")
    return 1 if any(c["status"] in {"error", "unverified"} or (c["status"] == "missing" and c["name"] != "LLM credentials") for c in checks) else 0


def _assets(args):
    _configure(args)
    from .reference_assets import import_reference, inventory
    if args.action == "import":
        print(json.dumps(import_reference(args.species[0], args.source), indent=2))
        return 0
    values = {s: inventory(s, full=args.action == "verify") for s in args.species}
    if args.models:
        from .model_assets import inventory as model_inventory
        values["models"] = model_inventory(args.species, full=args.action == "verify")
    if args.json:
        print(json.dumps(values, indent=2))
    else:
        for species, rows in values.items():
            print(species)
            for row in rows:
                print(f"  {row['status']:10} {row.get('bytes', 0):>14,}  {row['path']}")
    return 1 if args.action == "verify" and any(row["status"] not in {"verified", "optional"} for rows in values.values() for row in rows) else 0


def _run(argv):
    # Set locations before importing pipeline modules with cached configuration.
    locations = argparse.ArgumentParser(add_help=False)
    locations.add_argument("--data-dir", type=Path)
    locations.add_argument("--output-dir", type=Path)
    args, remainder = locations.parse_known_args(argv)
    _configure(args)
    if args.output_dir:
        os.environ["ASO_RUNS_DIR"] = str(args.output_dir.expanduser().resolve())
    from .runtime_binaries import refresh_runtime_path
    refresh_runtime_path()
    os.environ.setdefault("MODAL_ASO_SKIP_RUNTIME_INSTALL", "1")
    from .llm_provider import get_provider_config
    if not any(flag in remainder for flag in ("--help", "-h")):
        get_provider_config()
        print("Starting MODAL-ASO...", flush=True)
    from run_goal_intake import main as run
    run(argv=remainder)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if sys.platform == "win32" and argv and argv[0] not in {"--help", "-h", "--version"}:
            raise RuntimeError("On Windows, run MODAL-ASO inside WSL2: https://learn.microsoft.com/windows/wsl/install")
        if argv and argv[0] not in {"setup", "doctor", "assets", "--help", "-h", "--version"}:
            return _run(argv[1:] if argv[0] == "run" else argv)
        parser = argparse.ArgumentParser(prog="modal-aso", description='Design and evaluate ASOs. Run: modal-aso "your request"')
        parser.add_argument("--version", action="version", version=f"MODAL-ASO {__version__}")
        sub = parser.add_subparsers(dest="command")
        setup = sub.add_parser("setup", help="Install reusable tools, references and models")
        setup.add_argument("--data-dir", type=Path, help="Persistent data directory (saved for future sessions)")
        setup.add_argument("--species", nargs="+", choices=["human", "mouse"], default=["human"])
        setup.add_argument("--features", nargs="+", choices=["tools", "references", "splice", "oligoai", "all"], default=["tools", "references"])
        setup.set_defaults(func=_setup)
        doctor = sub.add_parser("doctor", help="Check the local installation")
        doctor.add_argument("--data-dir", type=Path)
        doctor.add_argument("--species", nargs="+", choices=["human", "mouse"], default=["human"])
        doctor.add_argument("--verify", action="store_true", help="Rehash reference files")
        doctor.add_argument("--json", action="store_true")
        doctor.set_defaults(func=_doctor)
        assets = sub.add_parser("assets", help="List, verify or import existing references")
        actions = assets.add_subparsers(dest="action", required=True)
        for name in ["list", "verify", "import"]:
            action = actions.add_parser(name)
            action.add_argument("--data-dir", type=Path)
            action.add_argument("--species", nargs="+" if name != "import" else 1, choices=["human", "mouse"], default=["human"], required=name == "import")
            if name == "import":
                action.add_argument("source", type=Path, help="Existing reference directory")
            else:
                action.add_argument("--json", action="store_true")
                action.add_argument("--models", action="store_true", help="Include installed model sources and checkpoints")
            action.set_defaults(func=_assets)
        args = parser.parse_args(argv)
        if not hasattr(args, "func"):
            parser.print_help()
            return 0
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"MODAL-ASO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
