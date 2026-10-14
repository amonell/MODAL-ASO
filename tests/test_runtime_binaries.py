from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aso_agents import runtime_binaries


def test_python_bin_dir_preserves_virtualenv_entrypoint(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime_binaries.sys, "executable", "/workspace/.venv/bin/python3")

    assert runtime_binaries._python_bin_dir() == Path("/workspace/.venv/bin")


def test_prepend_path_moves_existing_directory_to_front(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    virtualenv_bin = tmp_path / ".venv" / "bin"
    virtualenv_bin.mkdir(parents=True)
    base_python_bin = tmp_path / "uv-python" / "bin"
    base_python_bin.mkdir(parents=True)
    monkeypatch.setenv(
        "PATH",
        f"{base_python_bin}{runtime_binaries.os.pathsep}{virtualenv_bin}",
    )

    runtime_binaries._prepend_path(virtualenv_bin)

    assert runtime_binaries.os.environ["PATH"].split(runtime_binaries.os.pathsep) == [
        str(virtualenv_bin),
        str(base_python_bin),
    ]


@pytest.mark.real_bowtie_runtime
def test_ensure_bowtie_skips_install_when_binaries_exist(monkeypatch: Any, tmp_path: Path) -> None:
    bowtie = tmp_path / "bowtie"
    builder = tmp_path / "bowtie-build"
    bowtie.write_text("", encoding="utf-8")
    builder.write_text("", encoding="utf-8")
    bowtie.chmod(0o755)
    builder.chmod(0o755)

    def fake_which(name: str) -> str | None:
        if name == "bowtie":
            return str(bowtie)
        if name == "bowtie-build":
            return str(builder)
        return None

    monkeypatch.setattr(runtime_binaries.shutil, "which", fake_which)
    monkeypatch.setattr(runtime_binaries, "_refresh_homebrew_bowtie_path", lambda: None)

    result = runtime_binaries.ensure_bowtie_1(install=True)

    assert result["status"] == "ok"
    assert result["installed"] is False
    assert result["binaries"]["bowtie"] == str(bowtie)
    assert result["binaries"]["bowtie-build"] == str(builder)


@pytest.mark.real_bowtie_runtime
def test_ensure_bowtie_installs_via_homebrew_when_missing(monkeypatch: Any, tmp_path: Path) -> None:
    brew = tmp_path / "brew"
    bowtie = tmp_path / "bowtie"
    builder = tmp_path / "bowtie-build"
    installed = {"done": False}

    def fake_which(name: str) -> str | None:
        if name == "brew":
            return str(brew)
        if installed["done"] and name in {"bowtie", "bowtie-build"}:
            return str(bowtie if name == "bowtie" else builder)
        return None

    class DummyProc:
        returncode = 0

    def fake_run(command: list[str], **_kwargs: Any) -> DummyProc:
        installed["done"] = True
        bowtie.write_text("", encoding="utf-8")
        builder.write_text("", encoding="utf-8")
        return DummyProc()

    monkeypatch.setattr(runtime_binaries.shutil, "which", fake_which)
    monkeypatch.setattr(runtime_binaries, "_refresh_homebrew_bowtie_path", lambda: None)
    monkeypatch.setattr(runtime_binaries.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime_binaries.sys, "platform", "darwin")

    result = runtime_binaries.ensure_bowtie_1(install=True)

    assert result["status"] == "ok"
    assert result["installed"] is True
    assert result["command"] == [str(brew), "install", "brewsci/bio/bowtie"]


@pytest.mark.real_bowtie_runtime
def test_ensure_bowtie_reports_missing_when_install_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime_binaries.shutil, "which", lambda _name: None)
    monkeypatch.setattr(runtime_binaries, "_refresh_homebrew_bowtie_path", lambda: None)
    monkeypatch.setattr(runtime_binaries.sys, "platform", "darwin")

    result = runtime_binaries.ensure_bowtie_1(install=True)

    assert result["status"] == "missing"
    assert "brew install bowtie" in str(result["hint"]) or "brewsci/bio/bowtie" in str(result["hint"])
    assert "Bowtie 1 is required" in str(result["message"])


@pytest.mark.real_bowtie_runtime
def test_bowtie_install_hint_covers_windows(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime_binaries.sys, "platform", "win32")
    hint = runtime_binaries.bowtie_1_install_hint()
    assert "WSL2" in hint
    assert "apt-get install -y bowtie" in hint


@pytest.mark.real_runtime
def test_tiling_cli_hints_cover_macos_linux_and_windows(monkeypatch: Any) -> None:
    package = runtime_binaries.CliPackage(
        key="bedtools",
        display_name="BEDTools",
        binaries=("bedtools",),
        brew_formula="bedtools",
        apt_package="bedtools",
        conda_package="bedtools",
    )
    monkeypatch.setattr(runtime_binaries.sys, "platform", "darwin")
    darwin_hint = runtime_binaries.cli_package_install_hint(package)
    assert "brew install bedtools" in darwin_hint
    assert "apt-get install -y bedtools" in darwin_hint
    assert "WSL2" in darwin_hint

    monkeypatch.setattr(runtime_binaries.sys, "platform", "linux")
    linux_hint = runtime_binaries.cli_package_install_hint(package)
    assert "apt-get install -y bedtools" in linux_hint
    assert "conda install -y -c bioconda bedtools" in linux_hint

    monkeypatch.setattr(runtime_binaries.sys, "platform", "win32")
    windows_hint = runtime_binaries.cli_package_install_hint(package)
    assert "WSL2" in windows_hint
    assert "not supported natively on Windows" in windows_hint


@pytest.mark.real_runtime
def test_windows_native_skips_unix_binary_downloads_and_installs(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime_binaries.sys, "platform", "win32")
    assert runtime_binaries._subread_archive_url() is None
    assert runtime_binaries._ucsc_utilities_url("bedToBigBed") is None
    assert runtime_binaries._cli_install_command(runtime_binaries.TILING_CLI_PACKAGES[0]) is None
    assert runtime_binaries._install_command() is None
    result = runtime_binaries.ensure_run_runtime(install=True)
    assert result["status"] == "missing"
    assert "WSL2" in str(result["message"])


@pytest.mark.real_runtime
def test_ensure_snakemake_skips_install_when_present(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime_binaries, "snakemake_ready", lambda: True)
    monkeypatch.setattr(runtime_binaries, "snakemake_argv", lambda: ["/usr/bin/snakemake"])
    monkeypatch.setattr(runtime_binaries, "snakemake_executable", lambda: "/usr/bin/snakemake")

    result = runtime_binaries.ensure_snakemake(install=True)

    assert result["status"] == "ok"
    assert result["installed"] is False
    assert result["executable"] == "/usr/bin/snakemake"


@pytest.mark.real_runtime
def test_ensure_snakemake_installs_via_uv_when_missing(monkeypatch: Any, tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    ready = {"ok": False}

    def fake_which(name: str) -> str | None:
        if name == "uv":
            return str(uv)
        return None

    class DummyProc:
        returncode = 0

    def fake_run(command: list[str], **_kwargs: Any) -> DummyProc:
        ready["ok"] = True
        return DummyProc()

    monkeypatch.setattr(runtime_binaries, "snakemake_ready", lambda: ready["ok"])
    monkeypatch.setattr(runtime_binaries, "snakemake_argv", lambda: [str(tmp_path / "snakemake")])
    monkeypatch.setattr(runtime_binaries, "snakemake_executable", lambda: str(tmp_path / "snakemake"))
    monkeypatch.setattr(runtime_binaries.shutil, "which", fake_which)
    monkeypatch.setattr(runtime_binaries.subprocess, "run", fake_run)

    result = runtime_binaries.ensure_snakemake(install=True)

    assert result["status"] == "ok"
    assert result["installed"] is True
    assert result["command"][:3] == [str(uv), "pip", "install"]
    assert runtime_binaries.SNAKEMAKE_SPEC in result["command"]
    assert runtime_binaries.PULP_SPEC in result["command"]


@pytest.mark.real_runtime
def test_ensure_snakemake_reports_missing_when_install_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime_binaries, "snakemake_ready", lambda: False)
    monkeypatch.setattr(runtime_binaries, "snakemake_argv", lambda: [runtime_binaries.sys.executable, "-m", "snakemake"])
    monkeypatch.setattr(runtime_binaries, "snakemake_executable", lambda: None)
    monkeypatch.setattr(runtime_binaries.shutil, "which", lambda _name: None)

    result = runtime_binaries.ensure_snakemake(install=False)

    assert result["status"] == "missing"
    assert "uv pip install" in str(result["hint"])
    assert "Snakemake is required" in str(result["message"])


@pytest.mark.real_runtime
def test_bind_snakemake_command_rewrites_bare_snakemake(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime_binaries, "ensure_snakemake", lambda **_kwargs: {"status": "ok"})
    monkeypatch.setattr(
        runtime_binaries,
        "snakemake_argv",
        lambda: ["/venv/bin/python", "-m", "snakemake"],
    )

    rewritten = runtime_binaries.bind_snakemake_command(
        "cd /tmp/pipeline && PATH=/venv/bin:$PATH snakemake -s Snakefile --configfile /tmp/config.yaml -j 4"
    )

    assert rewritten.endswith(
        "/venv/bin/python -m snakemake -s Snakefile --configfile /tmp/config.yaml -j 4 --scheduler greedy"
    )
    assert "PATH=/venv/bin:$PATH /venv/bin/python -m snakemake" in rewritten


def test_bind_snakemake_command_preserves_explicit_scheduler(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime_binaries, "ensure_snakemake", lambda **_kwargs: {"status": "ok"})
    monkeypatch.setattr(runtime_binaries, "snakemake_argv", lambda: ["/venv/bin/snakemake"])

    rewritten = runtime_binaries.bind_snakemake_command(
        "snakemake -s Snakefile --configfile /tmp/config.yaml -j 4 --scheduler ilp"
    )

    assert rewritten.endswith("--scheduler ilp")
    assert "--scheduler greedy" not in rewritten


@pytest.mark.real_runtime
def test_ensure_run_runtime_reports_missing_snakemake(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        runtime_binaries,
        "ensure_bowtie_1",
        lambda **_kwargs: {"status": "ok", "installed": False},
    )
    monkeypatch.setattr(
        runtime_binaries,
        "ensure_snakemake",
        lambda **_kwargs: {"status": "missing", "message": "Snakemake is required"},
    )
    monkeypatch.setattr(
        runtime_binaries,
        "ensure_tiling_cli_tools",
        lambda **_kwargs: {"status": "ok", "missing": [], "tools": {}, "message": ""},
    )
    monkeypatch.setattr(runtime_binaries, "refresh_runtime_path", lambda: "/usr/bin")

    result = runtime_binaries.ensure_run_runtime(install=True)

    assert result["status"] == "missing"
    assert result["missing"] == ["snakemake"]
    assert "Snakemake is required" in str(result["message"])


@pytest.mark.real_runtime
def test_ensure_cli_package_skips_install_when_present(monkeypatch: Any, tmp_path: Path) -> None:
    bedtools = tmp_path / "bedtools"
    bedtools.write_text("", encoding="utf-8")
    bedtools.chmod(0o755)
    package = runtime_binaries.CliPackage(key="bedtools", display_name="BEDTools", binaries=("bedtools",))

    monkeypatch.setattr(runtime_binaries.shutil, "which", lambda name: str(bedtools) if name == "bedtools" else None)
    monkeypatch.setattr(runtime_binaries, "_refresh_homebrew_formula_path", lambda _formula: None)
    monkeypatch.setattr(runtime_binaries, "_prepend_path", lambda _directory: None)

    result = runtime_binaries.ensure_cli_package(package, install=True)

    assert result["status"] == "ok"
    assert result["installed"] is False
    assert result["binaries"]["bedtools"] == str(bedtools)


@pytest.mark.real_runtime
def test_ensure_cli_package_installs_via_homebrew(monkeypatch: Any, tmp_path: Path) -> None:
    brew = tmp_path / "brew"
    bedtools = tmp_path / "bedtools"
    installed = {"done": False}
    package = runtime_binaries.CliPackage(
        key="bedtools",
        display_name="BEDTools",
        binaries=("bedtools",),
        brew_formula="bedtools",
    )

    def fake_which(name: str) -> str | None:
        if name == "brew":
            return str(brew)
        if installed["done"] and name == "bedtools":
            return str(bedtools)
        return None

    class DummyProc:
        returncode = 0

    def fake_run(command: list[str], **_kwargs: Any) -> DummyProc:
        installed["done"] = True
        return DummyProc()

    monkeypatch.setattr(runtime_binaries.shutil, "which", fake_which)
    monkeypatch.setattr(runtime_binaries, "_refresh_homebrew_formula_path", lambda _formula: None)
    monkeypatch.setattr(runtime_binaries, "_prepend_path", lambda _directory: None)
    monkeypatch.setattr(runtime_binaries.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime_binaries.sys, "platform", "darwin")

    result = runtime_binaries.ensure_cli_package(package, install=True)

    assert result["status"] == "ok"
    assert result["installed"] is True
    assert result["command"] == [str(brew), "install", "bedtools"]


@pytest.mark.real_runtime
def test_subread_archive_url_uses_macos_arm64(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime_binaries.sys, "platform", "darwin")
    monkeypatch.setattr(runtime_binaries.platform, "machine", lambda: "arm64")
    url = runtime_binaries._subread_archive_url()
    assert url is not None
    assert "macOS-arm64" in url


@pytest.mark.real_runtime
def test_ensure_run_runtime_reports_missing_tiling_cli(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        runtime_binaries,
        "ensure_bowtie_1",
        lambda **_kwargs: {"status": "ok", "installed": False},
    )
    monkeypatch.setattr(
        runtime_binaries,
        "ensure_snakemake",
        lambda **_kwargs: {"status": "ok", "installed": False},
    )
    monkeypatch.setattr(
        runtime_binaries,
        "ensure_tiling_cli_tools",
        lambda **_kwargs: {
            "status": "missing",
            "missing": ["bedtools"],
            "tools": {},
            "message": "BEDTools is required",
        },
    )
    monkeypatch.setattr(runtime_binaries, "refresh_runtime_path", lambda: "/usr/bin")

    result = runtime_binaries.ensure_run_runtime(install=True)

    assert result["status"] == "missing"
    assert result["missing"] == ["bedtools"]
    assert "BEDTools is required" in str(result["message"])
