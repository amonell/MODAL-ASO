"""Install and locate runtime tools required before a design run starts.

Bowtie 1 (`bowtie` / `bowtie-build`) is a C++ aligner with no official PyPI
package, so this module installs it from Homebrew, apt, or conda/mamba.

Snakemake is a Python package. It is declared in `pyproject.toml`, and this
module also installs it into the active interpreter if it is still missing.

Steric tiling also needs BEDTools, samtools, featureCounts, RNAfold, and
bedToBigBed. Those are installed from Homebrew, apt, conda, or a UCSC download.
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


BOWTIE_1_BINARIES = ("bowtie", "bowtie-build")
SNAKEMAKE_SPEC = "snakemake>=7.32.4,<8"
# Snakemake 7.32.4 still calls pulp.list_solvers, which PuLP 2.8+ removed.
PULP_SPEC = "pulp>=2.3.1,<2.8"
SNAKEMAKE_INSTALL_SPECS = (SNAKEMAKE_SPEC, PULP_SPEC)


def native_windows() -> bool:
    return sys.platform == "win32"


def windows_wsl_hint(tool: str) -> str:
    return (
        f"{tool} is not supported natively on Windows. "
        "Install WSL2, clone/run MODAL-ASO inside Linux, then follow the Linux install instructions."
    )


def _auto_install_allowed(*env_flags: str) -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    flags = env_flags or ("MODAL_ASO_SKIP_BOWTIE_INSTALL",)
    for flag_name in (*flags, "MODAL_ASO_SKIP_RUNTIME_INSTALL"):
        flag = str(os.environ.get(flag_name) or "").strip().lower()
        if flag in {"1", "true", "yes"}:
            return False
    return True


def _python_bin_dir() -> Path:
    # Keep the virtualenv entrypoint path. Resolving its symlink points at uv's
    # bare base interpreter, whose bin directory lacks project dependencies.
    return Path(sys.executable).parent


def _prepend_path(directory: Path) -> None:
    if not directory.is_dir():
        return
    current = os.environ.get("PATH") or ""
    prefix = str(directory)
    parts = current.split(os.pathsep) if current else []
    # Move an existing entry to the front as well as adding a missing entry.
    # `uv run` commonly includes the virtualenv after uv's base Python bin.
    parts = [part for part in parts if part != prefix]
    os.environ["PATH"] = os.pathsep.join([prefix, *parts]) if parts else prefix


def runtime_bin_dir() -> Path:
    from .paths import data_root
    return data_root() / "runtime" / "bin"


def _refresh_homebrew_formula_path(formula: str) -> None:
    brew = shutil.which("brew")
    if not brew or not formula:
        return
    names = [formula, formula.split("/")[-1]]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        try:
            prefix = subprocess.check_output(
                [brew, "--prefix", name],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            continue
        if prefix:
            _prepend_path(Path(prefix) / "bin")
            return


def _refresh_homebrew_bowtie_path() -> None:
    _refresh_homebrew_formula_path("bowtie")


def refresh_runtime_path() -> str:
    """Put auto-installed tiling binaries on PATH for this process and children."""
    _prepend_path(runtime_bin_dir())
    _prepend_path(_python_bin_dir())
    for formula in ("bowtie", "bedtools", "samtools", "subread", "viennarna"):
        _refresh_homebrew_formula_path(formula)
    return os.environ.get("PATH") or ""


def bowtie_1_binaries() -> Dict[str, Optional[str]]:
    _prepend_path(runtime_bin_dir())
    _refresh_homebrew_bowtie_path()
    return {name: shutil.which(name) for name in BOWTIE_1_BINARIES}


def bowtie_1_ready() -> bool:
    found = bowtie_1_binaries()
    return all(found.get(name) for name in BOWTIE_1_BINARIES)


def bowtie_1_install_hint() -> str:
    if sys.platform == "darwin":
        return "Install Bowtie 1 with Homebrew: brew install brewsci/bio/bowtie"
    if sys.platform == "win32":
        return (
            "Bowtie 1 is not supported natively on Windows. "
            "Install WSL2, then inside Linux run: sudo apt-get install -y bowtie"
        )
    if shutil.which("apt-get"):
        return "Install Bowtie 1 with apt: sudo apt-get install -y bowtie"
    if shutil.which("conda") or shutil.which("mamba") or shutil.which("micromamba"):
        return "Install Bowtie 1 with conda: conda install -y -c bioconda bowtie"
    return (
        "Install Bowtie 1 (the `bowtie` and `bowtie-build` binaries). "
        "macOS: brew install brewsci/bio/bowtie. "
        "Debian/Ubuntu: sudo apt-get install -y bowtie. "
        "conda: conda install -y -c bioconda bowtie. "
        "Windows: use WSL2, then sudo apt-get install -y bowtie."
    )


def _install_command() -> Optional[List[str]]:
    if native_windows():
        return None
    brew = shutil.which("brew")
    if brew:
        # Homebrew core's `bowtie` cask is an unrelated music player.
        # Bowtie 1 (the aligner) lives in the brewsci/bio tap.
        formula = "brewsci/bio/bowtie" if sys.platform == "darwin" else "bowtie"
        return [brew, "install", formula]
    for condaish in ("micromamba", "mamba", "conda"):
        executable = shutil.which(condaish)
        if executable:
            return [executable, "install", "-y", "-c", "bioconda", "bowtie"]
    apt = shutil.which("apt-get")
    if apt:
        sudo = shutil.which("sudo")
        if sudo:
            return [sudo, "-n", apt, "install", "-y", "bowtie"]
        if os.geteuid() == 0:
            return [apt, "install", "-y", "bowtie"]
    return None


def _run_install(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    print(f"Installing Bowtie 1 with: {' '.join(command)}", flush=True)
    return subprocess.run(list(command), text=True)


def ensure_bowtie_1(*, install: Optional[bool] = None) -> Dict[str, Any]:
    """Return Bowtie 1 status, installing it when allowed and missing."""
    if bowtie_1_ready():
        return {
            "status": "ok",
            "installed": False,
            "binaries": bowtie_1_binaries(),
        }
    should_install = _auto_install_allowed("MODAL_ASO_SKIP_BOWTIE_INSTALL") if install is None else bool(install)
    command = _install_command() if should_install else None
    install_error = ""
    if command:
        try:
            proc = _run_install(command)
        except Exception as exc:
            install_error = str(exc)
            proc = None
        else:
            if proc.returncode != 0:
                install_error = f"install_exited_{proc.returncode}"
        _refresh_homebrew_bowtie_path()
        if bowtie_1_ready():
            return {
                "status": "ok",
                "installed": True,
                "command": list(command),
                "binaries": bowtie_1_binaries(),
            }
    hint = bowtie_1_install_hint()
    payload: Dict[str, Any] = {
        "status": "missing",
        "installed": False,
        "binaries": bowtie_1_binaries(),
        "hint": hint,
        "message": (
            "Bowtie 1 is required for tiling and off-target mapping, but "
            f"`bowtie` / `bowtie-build` were not found. {hint}"
        ),
    }
    if command:
        payload["command"] = list(command)
    if install_error:
        payload["install_error"] = install_error
    return payload


def _refresh_snakemake_path() -> None:
    _prepend_path(_python_bin_dir())


def _snakemake_works(argv: Sequence[str]) -> bool:
    try:
        proc = subprocess.run(
            [*list(argv), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return False
    return proc.returncode == 0


def snakemake_argv() -> List[str]:
    """Return an argv prefix that invokes Snakemake in the active environment."""
    _refresh_snakemake_path()
    found = shutil.which("snakemake")
    if found and _snakemake_works([found]):
        return [found]
    module_argv = [sys.executable, "-m", "snakemake"]
    if _snakemake_works(module_argv):
        return module_argv
    if found:
        return [found]
    return module_argv


def snakemake_executable() -> Optional[str]:
    argv = snakemake_argv()
    if argv and _snakemake_works(argv):
        return argv[0]
    return None


def snakemake_ready() -> bool:
    return bool(snakemake_executable())


def bind_snakemake_command(command: str) -> str:
    """Bind a stored tiling command to the environment's compatible Snakemake."""
    ensure_snakemake()
    prefix = " ".join(shlex.quote(part) for part in snakemake_argv())
    text = str(command or "")
    invocation = (
        r"(?:(?:'[^']+'|\"[^\"]+\"|\S+)\s+-m\s+snakemake|"
        r"(?:'[^']+'|\"[^\"]+\"|\S*snakemake))"
    )
    updated = re.sub(rf"(?<=\$PATH ){invocation}\b", prefix, text, count=1)
    if updated == text:
        updated = re.sub(r"(^|[;&|]\s*)snakemake\b", rf"\1{prefix}", text, count=1)
    # Snakemake 7's default ILP scheduler imports stopit, whose latest release
    # still imports the removed ``pkg_resources`` module.  These tiling DAGs are
    # small, so the built-in greedy scheduler is both sufficient and compatible
    # with current Setuptools releases.  Preserve an explicitly chosen scheduler.
    if "--scheduler" not in updated:
        updated = f"{updated} --scheduler greedy"
    return updated


def snakemake_install_hint() -> str:
    return (
        "Install Snakemake into this environment with "
        f"`uv pip install --python {sys.executable} {' '.join(SNAKEMAKE_INSTALL_SPECS)}` "
        "or `uv sync`."
    )


def _snakemake_install_command() -> List[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, *SNAKEMAKE_INSTALL_SPECS]
    return [sys.executable, "-m", "pip", "install", *SNAKEMAKE_INSTALL_SPECS]


def ensure_snakemake(*, install: Optional[bool] = None) -> Dict[str, Any]:
    """Return Snakemake status, installing it into the active interpreter when missing."""
    if snakemake_ready():
        return {
            "status": "ok",
            "installed": False,
            "argv": snakemake_argv(),
            "executable": snakemake_executable(),
        }
    should_install = (
        _auto_install_allowed("MODAL_ASO_SKIP_SNAKEMAKE_INSTALL") if install is None else bool(install)
    )
    command = _snakemake_install_command() if should_install else None
    install_error = ""
    if command:
        print(f"Installing Snakemake with: {' '.join(command)}", flush=True)
        try:
            proc = subprocess.run(list(command), text=True)
        except Exception as exc:
            install_error = str(exc)
            proc = None
        else:
            if proc.returncode != 0:
                install_error = f"install_exited_{proc.returncode}"
        _refresh_snakemake_path()
        if snakemake_ready():
            return {
                "status": "ok",
                "installed": True,
                "command": list(command),
                "argv": snakemake_argv(),
                "executable": snakemake_executable(),
            }
    hint = snakemake_install_hint()
    payload: Dict[str, Any] = {
        "status": "missing",
        "installed": False,
        "argv": snakemake_argv(),
        "executable": snakemake_executable(),
        "hint": hint,
        "message": (
            "Snakemake is required for steric tiling, but the `snakemake` command "
            f"was not found in this environment. {hint}"
        ),
    }
    if command:
        payload["command"] = list(command)
    if install_error:
        payload["install_error"] = install_error
    return payload


@dataclass(frozen=True)
class CliPackage:
    key: str
    display_name: str
    binaries: Tuple[str, ...]
    brew_formula: str = ""
    apt_package: str = ""
    conda_package: str = ""
    download_name: str = ""
    required_for: str = "steric tiling"


TILING_CLI_PACKAGES: Tuple[CliPackage, ...] = (
    CliPackage(
        key="bedtools",
        display_name="BEDTools",
        binaries=("bedtools",),
        brew_formula="bedtools",
        apt_package="bedtools",
        conda_package="bedtools",
        required_for="sequence tiling via pybedtools",
    ),
    CliPackage(
        key="samtools",
        display_name="samtools",
        binaries=("samtools",),
        brew_formula="samtools",
        apt_package="samtools",
        conda_package="samtools",
        required_for="secondary mapping BAM conversion",
    ),
    CliPackage(
        key="subread",
        display_name="featureCounts (Subread)",
        binaries=("featureCounts",),
        brew_formula="brewsci/bio/subread",
        apt_package="subread",
        conda_package="subread",
        required_for="secondary annotation assignment",
    ),
    CliPackage(
        key="viennarna",
        display_name="ViennaRNA (RNAfold)",
        binaries=("RNAfold",),
        brew_formula="brewsci/bio/viennarna",
        apt_package="viennarna",
        conda_package="viennarna",
        required_for="ASO secondary-structure folding",
    ),
    CliPackage(
        key="bedToBigBed",
        display_name="bedToBigBed",
        binaries=("bedToBigBed",),
        conda_package="ucsc-bedtobigbed",
        download_name="bedToBigBed",
        required_for="scored-tile bigBed export",
    ),
)


def _ucsc_utilities_url(name: str) -> Optional[str]:
    if native_windows():
        return None
    system = sys.platform
    machine = platform.machine().lower()
    if system == "darwin":
        folder = "macOSX.arm64" if machine in {"arm64", "aarch64"} else "macOSX.x86_64"
    elif machine in {"aarch64", "arm64"}:
        folder = "linux.aarch64"
    else:
        folder = "linux.x86_64"
    return f"https://hgdownload.soe.ucsc.edu/admin/exe/{folder}/{name}"


def _trust_brew_formula(formula: str) -> None:
    if "/" not in formula:
        return
    brew = shutil.which("brew")
    if not brew:
        return
    subprocess.run([brew, "trust", "--formula", formula], text=True, check=False)


def _subread_archive_url() -> Optional[str]:
    if native_windows():
        return None
    system = sys.platform
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        archive = "subread-2.1.1-macOS-arm64.tar.gz"
    elif system == "darwin":
        archive = "subread-2.1.1-macOS-x86_64.tar.gz"
    elif system.startswith("linux") and machine in {"x86_64", "amd64"}:
        archive = "subread-2.1.1-Linux-x86_64.tar.gz"
    else:
        return None
    return f"https://downloads.sourceforge.net/project/subread/subread-2.1.1/{archive}"


def _download_subread_binary() -> Optional[Path]:
    url = _subread_archive_url()
    if not url:
        return None
    destination = runtime_bin_dir() / "featureCounts"
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Subread (featureCounts) from {url}", flush=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "subread.tar.gz"
            with urllib.request.urlopen(url, timeout=120) as response:
                archive_path.write_bytes(response.read())
            with tarfile.open(archive_path, "r:gz") as archive:
                member = next(
                    (
                        item
                        for item in archive.getmembers()
                        if Path(item.name).name == "featureCounts" and item.isfile()
                    ),
                    None,
                )
                if member is None:
                    return None
                extracted = archive.extractfile(member)
                if extracted is None:
                    return None
                destination.write_bytes(extracted.read())
    except Exception:
        return None
    destination.chmod(0o755)
    return destination if destination.is_file() else None


def _cli_install_command(package: CliPackage) -> Optional[List[str]]:
    if native_windows():
        return None
    # brewsci/bio/subread 2.0.1 compiles with -msse2/-msse3 and fails on Apple Silicon.
    skip_brew = package.key == "subread" and sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    brew = None if skip_brew else shutil.which("brew")
    if brew and package.brew_formula:
        return [brew, "install", package.brew_formula]
    for condaish in ("micromamba", "mamba", "conda"):
        executable = shutil.which(condaish)
        if executable and package.conda_package:
            return [executable, "install", "-y", "-c", "bioconda", package.conda_package]
    apt = shutil.which("apt-get")
    if apt and package.apt_package:
        sudo = shutil.which("sudo")
        if sudo:
            return [sudo, "-n", apt, "install", "-y", package.apt_package]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            return [apt, "install", "-y", package.apt_package]
    return None


def _download_ucsc_binary(name: str) -> Optional[Path]:
    if native_windows():
        return None
    url = _ucsc_utilities_url(name)
    if not url:
        return None
    destination = runtime_bin_dir() / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} from {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    except Exception:
        return None
    destination.chmod(0o755)
    return destination if destination.is_file() else None


def cli_package_binaries(package: CliPackage) -> Dict[str, Optional[str]]:
    if package.brew_formula:
        _refresh_homebrew_formula_path(package.brew_formula)
    _prepend_path(runtime_bin_dir())
    return {name: shutil.which(name) for name in package.binaries}


def cli_package_ready(package: CliPackage) -> bool:
    found = cli_package_binaries(package)
    return all(found.get(name) for name in package.binaries)


def cli_package_install_hint(package: CliPackage) -> str:
    if native_windows():
        return windows_wsl_hint(package.display_name)
    parts: List[str] = []
    if sys.platform == "darwin" and package.brew_formula:
        if package.key == "subread" and platform.machine().lower() in {"arm64", "aarch64"}:
            parts.append(
                "macOS Apple Silicon: preflight downloads the official Subread arm64 binary "
                "(Homebrew's subread formula does not build on ARM)."
            )
        else:
            parts.append(f"macOS: brew install {package.brew_formula}")
    if package.apt_package:
        parts.append(f"Linux (Debian/Ubuntu): sudo apt-get install -y {package.apt_package}")
    if package.conda_package:
        parts.append(f"Linux (conda/mamba): conda install -y -c bioconda {package.conda_package}")
    if package.key == "subread":
        url = _subread_archive_url()
        if url:
            parts.append(f"Or download featureCounts from {url}.")
    if package.download_name:
        url = _ucsc_utilities_url(package.download_name)
        if url:
            parts.append(f"Or download {package.display_name} from {url}.")
    parts.append("Windows: use WSL2, then follow the Linux instructions.")
    return " ".join(parts) if parts else f"Install {package.display_name} ({', '.join(package.binaries)}) and put it on PATH."


def ensure_cli_package(package: CliPackage, *, install: Optional[bool] = None) -> Dict[str, Any]:
    if cli_package_ready(package):
        return {
            "status": "ok",
            "installed": False,
            "key": package.key,
            "binaries": cli_package_binaries(package),
        }
    should_install = (
        _auto_install_allowed("MODAL_ASO_SKIP_TILING_CLI_INSTALL") if install is None else bool(install)
    )
    command = _cli_install_command(package) if should_install else None
    install_error = ""
    if command and command[0] and Path(command[0]).name == "brew":
        _trust_brew_formula(package.brew_formula)
    if command:
        print(f"Installing {package.display_name} with: {' '.join(command)}", flush=True)
        try:
            proc = subprocess.run(list(command), text=True)
        except Exception as exc:
            install_error = str(exc)
            proc = None
        else:
            if proc.returncode != 0:
                install_error = f"install_exited_{proc.returncode}"
        if package.brew_formula:
            _refresh_homebrew_formula_path(package.brew_formula)
        if cli_package_ready(package):
            return {
                "status": "ok",
                "installed": True,
                "key": package.key,
                "command": list(command),
                "binaries": cli_package_binaries(package),
            }
    if should_install and package.key == "subread" and not cli_package_ready(package):
        downloaded = _download_subread_binary()
        _prepend_path(runtime_bin_dir())
        if downloaded and cli_package_ready(package):
            return {
                "status": "ok",
                "installed": True,
                "key": package.key,
                "command": ["download", str(downloaded)],
                "binaries": cli_package_binaries(package),
            }
    if should_install and package.download_name and not cli_package_ready(package):
        downloaded = _download_ucsc_binary(package.download_name)
        _prepend_path(runtime_bin_dir())
        if downloaded and cli_package_ready(package):
            return {
                "status": "ok",
                "installed": True,
                "key": package.key,
                "command": ["download", str(downloaded)],
                "binaries": cli_package_binaries(package),
            }
    hint = cli_package_install_hint(package)
    payload: Dict[str, Any] = {
        "status": "missing",
        "installed": False,
        "key": package.key,
        "binaries": cli_package_binaries(package),
        "hint": hint,
        "message": (
            f"{package.display_name} is required for {package.required_for}, but "
            f"{' / '.join(package.binaries)} were not found. {hint}"
        ),
    }
    if command:
        payload["command"] = list(command)
    if install_error:
        payload["install_error"] = install_error
    return payload


def tiling_cli_packages() -> Tuple[CliPackage, ...]:
    return TILING_CLI_PACKAGES


def tiling_cli_missing() -> List[str]:
    refresh_runtime_path()
    return [package.key for package in TILING_CLI_PACKAGES if not cli_package_ready(package)]


def tiling_cli_tools_ready() -> bool:
    return not tiling_cli_missing()


def tiling_cli_install_hint() -> str:
    missing = [package for package in TILING_CLI_PACKAGES if not cli_package_ready(package)]
    if not missing:
        return "Steric tiling CLI tools are installed."
    return " ".join(cli_package_install_hint(package) for package in missing)


def ensure_tiling_cli_tools(*, install: Optional[bool] = None) -> Dict[str, Any]:
    """Install BEDTools, samtools, featureCounts, RNAfold, and bedToBigBed."""
    refresh_runtime_path()
    tools: Dict[str, Dict[str, Any]] = {}
    for package in TILING_CLI_PACKAGES:
        tools[package.key] = ensure_cli_package(package, install=install)
    refresh_runtime_path()
    missing = [
        key
        for key, payload in tools.items()
        if str(payload.get("status") or "").strip().lower() != "ok"
    ]
    messages = [
        str(tools[key].get("message") or "")
        for key in missing
        if str(tools[key].get("message") or "").strip()
    ]
    return {
        "status": "ok" if not missing else "missing",
        "missing": missing,
        "tools": tools,
        "message": " ".join(messages),
    }


def ensure_run_runtime(*, install: Optional[bool] = None) -> Dict[str, Any]:
    """Install Bowtie 1, Snakemake, and steric tiling CLI tools before a run starts."""
    if native_windows():
        message = (
            "MODAL-ASO design runs are not supported natively on Windows. "
            "Install WSL2, clone the repo inside Linux, then run `uv sync` and "
            "`uv run python -m aso_agents.runtime_binaries`."
        )
        return {
            "status": "missing",
            "missing": ["windows_native"],
            "bowtie": {"status": "missing", "message": windows_wsl_hint("Bowtie 1")},
            "snakemake": {"status": "missing", "message": windows_wsl_hint("Snakemake tiling")},
            "tiling_cli": {"status": "missing", "missing": ["windows_native"], "message": message},
            "message": message,
        }
    refresh_runtime_path()
    bowtie = ensure_bowtie_1(install=install)
    snakemake = ensure_snakemake(install=install)
    tiling = ensure_tiling_cli_tools(install=install)
    missing = [
        name
        for name, payload in (("bowtie", bowtie), ("snakemake", snakemake))
        if str(payload.get("status") or "").strip().lower() != "ok"
    ]
    missing.extend(str(item) for item in (tiling.get("missing") or []) if str(item).strip())
    messages = []
    if "bowtie" in missing:
        messages.append(str(bowtie.get("message") or ""))
    if "snakemake" in missing:
        messages.append(str(snakemake.get("message") or ""))
    if tiling.get("message") and str(tiling.get("status") or "") != "ok":
        messages.append(str(tiling.get("message") or ""))
    return {
        "status": "ok" if not missing else "missing",
        "missing": missing,
        "bowtie": bowtie,
        "snakemake": snakemake,
        "tiling_cli": tiling,
        "message": " ".join(item for item in messages if item.strip()),
    }


def main() -> int:
    result = ensure_run_runtime(install=True)
    bowtie = result.get("bowtie") if isinstance(result.get("bowtie"), dict) else {}
    snakemake = result.get("snakemake") if isinstance(result.get("snakemake"), dict) else {}
    tiling = result.get("tiling_cli") if isinstance(result.get("tiling_cli"), dict) else {}
    if bowtie.get("status") == "ok":
        binaries = bowtie.get("binaries") if isinstance(bowtie.get("binaries"), dict) else {}
        action = "Installed" if bowtie.get("installed") else "Found"
        print(f"{action} Bowtie 1: {binaries.get('bowtie') or 'bowtie'} ; {binaries.get('bowtie-build') or 'bowtie-build'}")
    else:
        print(str(bowtie.get("message") or bowtie.get("hint") or "Bowtie 1 is not installed."), file=sys.stderr)
    if snakemake.get("status") == "ok":
        action = "Installed" if snakemake.get("installed") else "Found"
        print(f"{action} Snakemake: {' '.join(snakemake.get('argv') or ['snakemake'])}")
    else:
        print(str(snakemake.get("message") or snakemake.get("hint") or "Snakemake is not installed."), file=sys.stderr)
    tools = tiling.get("tools") if isinstance(tiling.get("tools"), dict) else {}
    for key, payload in tools.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "ok":
            action = "Installed" if payload.get("installed") else "Found"
            binaries = payload.get("binaries") if isinstance(payload.get("binaries"), dict) else {}
            found = ", ".join(str(path) for path in binaries.values() if path)
            print(f"{action} {key}: {found or key}")
        else:
            print(str(payload.get("message") or payload.get("hint") or f"{key} is not installed."), file=sys.stderr)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
