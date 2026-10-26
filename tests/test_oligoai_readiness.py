"""Installed assets must stay discoverable and incomplete runtimes fail early."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from aso_agents import cli, oligoai, oligoai_readiness as ready, oligoai_runtime, paths


@pytest.fixture
def existing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_ASO_CONFIG_FILE", str(tmp_path / "settings/config.json"))
    monkeypatch.setenv("MODAL_ASO_DATA_DIR", str(tmp_path / "other data"))
    monkeypatch.setenv("MODAL_ASO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ASO_OLIGOAI_ROOT", raising=False)
    root = tmp_path / "previous installation"
    (root / "venv/bin").mkdir(parents=True)
    (root / "venv/bin/python").symlink_to(sys.executable)
    source = root / "source"
    source.mkdir()
    (source / "model.py").write_text("# neutral source fixture\n")
    for command in [
        ["init", str(source)],
        ["-C", str(source), "add", "model.py"],
        ["-C", str(source), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-m", "neutral model fixture"],
    ]:
        subprocess.run(["git", *command], check=True, capture_output=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(oligoai, "SOURCE_REVISION", revision)
    content = b"neutral checkpoint fixture"
    (root / oligoai.MODEL_FILE).write_bytes(content)
    monkeypatch.setattr(oligoai, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    paths.save_settings({"reference_dir": "preserve references", "user_preference": "keep"})
    return root


def test_register_existing_survives_new_shell_working_directory_and_data_root(existing, tmp_path, monkeypatch):
    monkeypatch.setattr(oligoai_runtime, "_probe", lambda *_: pytest.fail("Registration must not use the GPU"))
    monkeypatch.setattr(oligoai_runtime, "download", lambda *_a, **_k: pytest.fail("No download permitted"))
    result = ready.register_existing(existing)
    assert result["registered"] and result["assets_verified"]
    assert not result["runtime_checked"] and result["status"] == "installed"
    assert paths.settings()["reference_dir"] == "preserve references"
    assert paths.settings()["user_preference"] == "keep"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
               MODAL_ASO_DATA_DIR=str(tmp_path / "different data"))
    env.pop("ASO_OLIGOAI_ROOT", None)
    actual = subprocess.check_output(
        [sys.executable, "-c", "from aso_agents.paths import oligoai_root; print(oligoai_root())"],
        cwd=tmp_path, env=env, text=True).strip()
    assert Path(actual) == existing


def test_cli_imports_an_existing_install_without_running_a_model(existing, capsys):
    assert cli.main(["assets", "import-model", "oligoai", str(existing)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["root"] == str(existing)
    assert result["registered"] and not result["runtime_checked"]
    assert paths.oligoai_root() == existing


def test_explicit_environment_override_does_not_silently_use_another_install(existing, tmp_path, monkeypatch):
    ready.register_existing(existing)
    missing = tmp_path / "invisible in this container"
    monkeypatch.setenv("ASO_OLIGOAI_ROOT", str(missing))
    assert paths.oligoai_root() == missing
    result = ready.inspect_installation()
    assert result["status"] == "missing" and str(missing) in result["detail"]
    assert "container or sandbox" in result["detail"]


@pytest.mark.parametrize("component", ["venv/bin/python", "source/model.py", "OligoAI_11_09_25.ckpt"])
def test_missing_or_modified_components_do_not_replace_registered_preferences(existing, component):
    paths.save_settings({"oligoai_root": "/previous/configured/root"})
    path = existing / component
    if component.endswith("model.py"):
        path.write_text("# changed pinned source\n")
    else:
        path.unlink()
    with pytest.raises(RuntimeError):
        ready.register_existing(existing)
    assert paths.settings()["oligoai_root"] == "/previous/configured/root"
    assert paths.settings()["user_preference"] == "keep"


def test_checkpoint_corruption_is_not_reported_as_not_installed(existing):
    (existing / oligoai.MODEL_FILE).write_bytes(b"wrong checkpoint")
    with pytest.raises(RuntimeError, match="checkpoint checksum mismatch"):
        ready.register_existing(existing)
    assert "oligoai_root" not in paths.settings()


def test_broken_interpreter_symlink_reports_the_actual_missing_path(existing):
    python = existing / "venv/bin/python"
    python.unlink()
    python.symlink_to(existing / "not-visible/python")
    result = ready.inspect_installation(existing)
    assert result["status"] == "incomplete"
    assert str(python) in result["detail"]
    assert "Python interpreter" in result["detail"]


def test_complete_assets_do_not_claim_current_gpu_readiness(existing):
    result = ready.inspect_installation(existing, verify=True)
    assert result["status"] == "installed"
    assert result["assets_verified"] and not result["runtime_checked"]
    assert "has not been tested" in result["detail"]


def test_explicit_preflight_checks_the_isolated_runtime_and_reports_cuda_failure(existing, monkeypatch):
    ready.register_existing(existing)
    def unavailable(python, source):
        assert python == existing / "venv/bin/python"
        assert source == existing / "source"
        raise subprocess.CalledProcessError(1, ["private probe"], stderr="Traceback\nAssertionError: CUDA is unavailable\n")
    monkeypatch.setattr(oligoai_runtime, "_probe", unavailable)
    with pytest.raises(RuntimeError) as error:
        ready.require_oligoai()
    message = str(error.value)
    assert "is installed at" in message
    assert "CUDA is unavailable" in message
    assert "not installed" not in message
    assert "private probe" not in message and "Traceback" not in message


def test_required_runtime_success_is_reported_only_after_probe(existing, monkeypatch):
    ready.register_existing(existing)
    seen = []
    monkeypatch.setattr(oligoai_runtime, "_probe", lambda *args: seen.append(args) or {"gpu": "test-only"})
    result = ready.require_oligoai()
    assert seen == [(existing / "venv/bin/python", existing / "source")]
    assert result["status"] == "ready" and result["runtime_checked"]


def test_required_runtime_fails_before_any_model_or_workflow_call(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ASO_OLIGOAI_ROOT", str(tmp_path / "not mounted"))
    import run_goal_intake
    from aso_agents import llm_provider
    monkeypatch.setattr(llm_provider, "get_provider_config", lambda *_a, **_k: pytest.fail("No LLM access before preflight"))
    monkeypatch.setattr(run_goal_intake, "main", lambda *_a, **_k: pytest.fail("No workflow before preflight"))
    assert cli.main(["--require-model", "oligoai", "A neutral request"]) == 1
    assert str(tmp_path / "not mounted") in capsys.readouterr().err


def test_required_runtime_flag_is_consumed_without_changing_request(monkeypatch):
    import run_goal_intake
    order = []
    monkeypatch.setattr(ready, "require_oligoai", lambda: order.append("ready") or {})
    def run(*, argv, result_callback):
        assert order == ["ready"]
        assert argv == ["A neutral request"]
        result_callback({"status": "run_complete"})
    monkeypatch.setattr(run_goal_intake, "main", run)
    assert cli.main(["--require-model", "oligoai", "A neutral request"]) == 0


def test_request_help_never_probes_gpu_or_requires_installed_models(monkeypatch):
    import run_goal_intake
    monkeypatch.setattr(ready, "require_oligoai", lambda: pytest.fail("Help must not probe the GPU"))
    monkeypatch.setattr(run_goal_intake, "main", lambda **_: None)
    assert cli.main(["--require-model", "oligoai", "--help"]) == 0


@pytest.mark.parametrize("required,partial,expected", [(False,False,0),(False,True,1),(True,False,1)])
def test_doctor_distinguishes_absent_optional_models_from_broken_or_required_models(tmp_path, monkeypatch, capsys, required, partial, expected):
    from aso_agents import reference_assets, model_assets, runtime_binaries, llm_provider
    root = tmp_path / "model"
    monkeypatch.setenv("ASO_OLIGOAI_ROOT", str(root))
    if partial:
        root.mkdir()
        (root / oligoai.MODEL_FILE).write_text("checkpoint only")
    monkeypatch.setattr(runtime_binaries,"refresh_runtime_path",lambda:None)
    monkeypatch.setattr(cli.shutil,"which",lambda _:"/test/tool")
    monkeypatch.setattr(reference_assets,"inventory",lambda *_a,**_k:[])
    monkeypatch.setattr(model_assets,"inventory",lambda *_a,**_k:[])
    monkeypatch.setattr(llm_provider,"get_provider_config",lambda **_:SimpleNamespace(api_key="test-only",name="test"))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
    args=["doctor","--json"]+(["--require-model","oligoai"] if required else [])
    assert cli.main(args)==expected
    output=json.loads(capsys.readouterr().out)
    checks={item["name"]:item for item in output["checks"]}
    if required:
        assert checks["OligoAI runtime"]["status"]=="error"
        assert str(root) in checks["OligoAI runtime"]["detail"]
    else:
        assert checks["OligoAI runtime files"]["status"]==("error" if partial else "optional")


@pytest.mark.parametrize("value", [[], 12, ""])
def test_invalid_registered_directory_is_actionable(tmp_path, monkeypatch, value):
    monkeypatch.delenv("ASO_OLIGOAI_ROOT", raising=False)
    monkeypatch.setenv("MODAL_ASO_CONFIG_FILE", str(tmp_path / "config.json"))
    paths.save_settings({"oligoai_root": value})
    with pytest.raises(ValueError, match="OligoAI directory"):
        paths.oligoai_root()


@pytest.mark.parametrize("warm", [True, False])
def test_setup_remembers_custom_root_after_success_and_preserves_settings(existing, monkeypatch, warm):
    monkeypatch.setattr(oligoai_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(oligoai_runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(oligoai_runtime, "install_source", lambda *_: {})
    monkeypatch.setattr(oligoai_runtime, "download", lambda *_a, **_k: {})
    monkeypatch.setattr(oligoai_runtime, "_probe", lambda *_: {"gpu": "test-only"})
    monkeypatch.setattr(oligoai_runtime.subprocess, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(oligoai_runtime.subprocess, "check_output", lambda *_a, **_k: "[]")
    identity = {
        "source_revision": oligoai_runtime.SOURCE_REVISION, "model_revision": oligoai_runtime.MODEL_REVISION,
        "model_sha256": oligoai_runtime.MODEL_SHA256,
        "requirements_sha256": oligoai_runtime.sha256(paths.resource_root() / "requirements/oligoai.txt"),
        "flash_sha256": oligoai_runtime.FLASH_SHA256, "environment_schema": 1,
    }
    if warm:
        (existing / "installation.json").write_text(json.dumps({"identity": identity}))
    assert oligoai_runtime.install(existing) == existing
    assert paths.settings()["oligoai_root"] == str(existing)
    assert paths.settings()["user_preference"] == "keep"


def test_failed_setup_does_not_register_an_unusable_environment(existing, monkeypatch):
    monkeypatch.setattr(oligoai_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(oligoai_runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(oligoai_runtime, "install_source", lambda *_: {})
    monkeypatch.setattr(oligoai_runtime, "download", lambda *_a, **_k: {})
    monkeypatch.setattr(oligoai_runtime.subprocess, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(oligoai_runtime, "_probe", lambda *_: (_ for _ in ()).throw(RuntimeError("CUDA unavailable")))
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        oligoai_runtime.install(existing)
    assert "oligoai_root" not in paths.settings()
