from __future__ import annotations

from pathlib import Path
from typing import Any

from aso_agents import pipeline_tools


def test_missing_reference_preflight_returns_clarification_context(monkeypatch, tmp_path: Path) -> None:
    missing_fasta = tmp_path / "missing.fa"

    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
            "structured_goal": {
                "species": "human",
                "intent": "knockdown",
                "design_programs": [{"modality": "gapmer"}],
            },
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: False)
    monkeypatch.setattr(
        pipeline_tools,
        "_required_tiling_reference_files",
        lambda _species: [str(missing_fasta)],
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("human genome FASTA", [str(missing_fasta)])],
    )

    preflight = pipeline_tools._preflight_reference_requirements(str(tmp_path))
    questions, reason, context = pipeline_tools._reference_preflight_questions(preflight)

    assert preflight["ok"] is False
    assert preflight["missing_files"] == [str(missing_fasta)]
    assert preflight["missing_requirements"] == ["human genome FASTA"]
    assert "reference_data_missing" in reason
    assert len(questions) == 1
    assert "download to default" in questions[0]
    assert "skip" in questions[0]
    assert "Bowtie 1" in context


def test_static_reference_download_accepts_custom_data_dir(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    refs_dir = tmp_path / "refs"
    state: dict[str, Any] = {
        "run_dir": str(run_dir),
        "reference_download_request": {
            "species": ["human"],
            "download_dir": str(refs_dir),
        },
    }
    commands: list[list[str]] = []

    class DummyProc:
        returncode = 0

    def fake_prepare_execution_stage(**_kwargs: Any) -> dict[str, Any]:
        return {"status": "ok", "state": state, "run_dir": str(run_dir)}

    def fake_run(cmd: list[str], **_kwargs: Any) -> DummyProc:
        commands.append(cmd)
        return DummyProc()

    def fake_promote_atomic_status(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "state_path": kwargs.get("state_path"),
            "run_dir": str(run_dir),
            "reference_download_results": kwargs.get("extra", {}).get("reference_download_results"),
        }

    monkeypatch.setattr(pipeline_tools, "_prepare_execution_stage", fake_prepare_execution_stage)
    monkeypatch.setattr(pipeline_tools.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline_tools, "_save_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_tools, "_promote_atomic_status", fake_promote_atomic_status)
    monkeypatch.setattr(
        pipeline_tools,
        "_apply_reference_data_root_choice",
        lambda root: str(Path(root).expanduser().resolve()),
    )

    result = pipeline_tools.tool_download_reference_assets(str(tmp_path / "state.json"))

    assert result["status"] == "ok"
    assert commands
    assert "--data-dir" in commands[0]
    assert commands[0][commands[0].index("--data-dir") + 1] == str(refs_dir.resolve())
    assert state["reference_data_root"] == str(refs_dir.resolve())


def test_mouse_splice_preflight_passes_with_explicit_local_assets(monkeypatch, tmp_path: Path) -> None:
    reference = tmp_path / "mouse.fa"
    script = tmp_path / "run_openspliceai_eval.py"
    model = tmp_path / "mouse_model.pt"
    for path in (reference, script, model):
        path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
            "structured_goal": {
                "species": "mouse",
                "intent": "splice_modulation",
                "run_splice_eval": True,
                "design_programs": [
                    {"modality": "steric", "intent": "splice_modulation", "run_splice_eval": True}
                ],
            },
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: True)
    monkeypatch.setattr(pipeline_tools, "_required_tiling_reference_files", lambda _species: [str(reference)])
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("mouse genome FASTA", [str(reference)])],
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_required_openspliceai_files",
        lambda _model_path=None: [str(script), str(model)],
    )
    monkeypatch.setattr(pipeline_tools, "_openspliceai_runtime_missing_reason", lambda: "")

    preflight = pipeline_tools._preflight_reference_requirements(str(tmp_path))

    assert preflight["ok"] is True
    assert preflight["missing_requirements"] == []
    assert preflight["missing_files"] == []
    assert preflight["splice_eval_backend"] == "openspliceai"
    assert "OpenSpliceAI model:" in preflight["hint"]


def test_missing_human_splice_eval_assets_generate_clarification_question(monkeypatch, tmp_path: Path) -> None:
    reference = tmp_path / "human.fa"
    missing_checkpoint = tmp_path / "SpTransformer_pytorch.ckpt"
    missing_hg38 = tmp_path / "hg38.fa"
    reference.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
                "structured_goal": {
                    "species": "human",
                    "target_context": "human motor neurons in SMA model",
                    "intent": "splice_modulation",
                    "run_splice_eval": True,
                    "design_programs": [
                    {"modality": "steric", "intent": "splice_modulation", "run_splice_eval": True}
                ],
            },
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "resolve_ensembl_species", lambda _species: {"status": "resolved"})
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: True)
    monkeypatch.setattr(pipeline_tools, "resolve_splice_eval_backend", lambda _species: {"backend": "splicetransformer"})
    monkeypatch.setattr(pipeline_tools, "_required_tiling_reference_files", lambda _species: [str(reference)])
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("human genome FASTA", [str(reference)])],
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_required_human_splice_eval_files",
        lambda: [str(missing_checkpoint), str(missing_hg38)],
    )

    preflight = pipeline_tools._preflight_reference_requirements(str(tmp_path))
    questions, reason, context = pipeline_tools._reference_preflight_questions(preflight)

    assert preflight["ok"] is False
    assert preflight["missing_requirements"] == ["SpliceTransformer submodule/weights/reference FASTA"]
    assert questions
    assert "SpliceTransformer submodule/weights/reference FASTA" in questions[0]
    assert str(missing_checkpoint) in questions[0]
    assert "install splice assets" in questions[0]
    assert "skip this species/task" in questions[0]
    assert "reference_data_missing" in reason
    assert "SpliceTransformer weights" in context


def test_reference_preflight_tool_pauses_for_missing_splice_assets(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "state" / "run_state.json"
    reference = tmp_path / "human.fa"
    missing_checkpoint = tmp_path / "SpTransformer_pytorch.ckpt"
    reference.write_text("placeholder", encoding="utf-8")
    state = {
        "version": "2",
        "status": "resolution_ready",
        "phase": "resolution_ready",
        "request": "Design human SMN2 splice-switching ASOs.",
        "flags": {},
        "run_dir": str(run_dir),
        "artifacts": {},
        "events": [],
        "last_error": None,
        "pending_questions": [],
        "clarification": {"isoform_scope_confirmed": False},
        "clarification_qa_history": [],
        "force_rerun_stages": [],
        "stage_attempts": {},
    }
    pipeline_tools._save_state(str(state_path), state, snapshot_reason="unit_seed")

    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
            "structured_goal": {
                "species": "human",
                "target_context": "human motor neurons in SMA model",
                "intent": "splice_modulation",
                "run_splice_eval": True,
                "design_programs": [
                    {"modality": "steric", "intent": "splice_modulation", "run_splice_eval": True}
                ],
            },
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "resolve_ensembl_species", lambda _species: {"status": "resolved"})
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: True)
    monkeypatch.setattr(pipeline_tools, "resolve_splice_eval_backend", lambda _species: {"backend": "splicetransformer"})
    monkeypatch.setattr(pipeline_tools, "_required_tiling_reference_files", lambda _species: [str(reference)])
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("human genome FASTA", [str(reference)])],
    )
    monkeypatch.setattr(pipeline_tools, "_required_human_splice_eval_files", lambda: [str(missing_checkpoint)])

    result = pipeline_tools.tool_reference_preflight(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok"
    assert result["needs_user_input"] is True
    assert result["clarification_kind"] == "reference_preflight"
    assert result["suggested_answers"] == [
        "install splice assets",
        "skip this species/task",
        "stop",
    ]
    assert saved_state["status"] == "needs_user_input"
    assert saved_state["phase"] == "awaiting_clarification"
    assert saved_state["pending_questions"]
    assert "SpliceTransformer" in saved_state["pending_questions"][0]


def test_reference_preflight_reuses_cached_ok_without_rechecking(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "state" / "run_state.json"
    cached_preflight = {
        "ok": True,
        "species": "human",
        "missing_files": [],
        "missing_requirements": [],
    }
    state = {
        "version": "2",
        "status": "resolution_ready",
        "phase": "resolution_ready",
        "request": "Design human CDKN1B 3' UTR ASOs.",
        "flags": {},
        "run_dir": str(run_dir),
        "artifacts": {},
        "events": [],
        "last_error": None,
        "pending_questions": [],
        "clarification": {"isoform_scope_confirmed": False},
        "clarification_qa_history": [],
        "force_rerun_stages": [],
        "stage_attempts": {},
        "preflight": cached_preflight,
    }
    pipeline_tools._save_state(str(state_path), state, snapshot_reason="unit_seed")

    def fail_if_rechecked(_run_dir: str) -> dict[str, Any]:
        raise AssertionError("cached successful preflight should not be rechecked")

    monkeypatch.setattr(pipeline_tools, "_preflight_reference_requirements", fail_if_rechecked)

    result = pipeline_tools.tool_reference_preflight(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok"
    assert result["cached"] is True
    assert result["preflight"] == cached_preflight
    assert saved_state["status"] == "preflight_ready"
    assert saved_state["phase"] == "preflight_ready"
    assert saved_state["preflight"] == cached_preflight


def test_reference_preflight_force_rerun_persists_recomputed_ok(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "state" / "run_state.json"
    reference = tmp_path / "human.fa"
    reference.write_text("placeholder", encoding="utf-8")
    stale_missing = tmp_path / "missing_before_download.fa"
    state = {
        "version": "2",
        "status": "reference_assets_downloaded",
        "phase": "reference_assets_downloaded",
        "request": "Design human CDKN1B 3' UTR ASOs.",
        "flags": {},
        "run_dir": str(run_dir),
        "artifacts": {},
        "events": [],
        "last_error": None,
        "pending_questions": [],
        "clarification": {"isoform_scope_confirmed": False},
        "clarification_qa_history": [
            {"question": "Reference assets are missing for human.", "answer": "download to default"}
        ],
        "force_rerun_stages": [],
        "stage_attempts": {},
        "preflight": {
            "ok": False,
            "missing_files": [str(stale_missing)],
            "missing_requirements": ["human genome FASTA"],
        },
    }
    pipeline_tools._save_state(str(state_path), state, snapshot_reason="unit_seed")

    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
                "structured_goal": {
                    "species": "human",
                    "target_context": "human dermal fibroblasts",
                    "intent": "knockdown",
                    "design_programs": [{"modality": "gapmer"}],
                },
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "resolve_ensembl_species", lambda _species: {"status": "resolved"})
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: False)
    monkeypatch.setattr(pipeline_tools, "_required_tiling_reference_files", lambda _species: [str(reference)])
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("human genome FASTA", [str(reference)])],
    )

    result = pipeline_tools.tool_reference_preflight(
        str(state_path),
        force_rerun=True,
        issue_summary="Reference assets were downloaded and need validation before continuing.",
    )
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok"
    assert result["preflight"]["ok"] is True
    assert result["preflight"]["missing_files"] == []
    assert saved_state["status"] == "preflight_ready"
    assert saved_state["phase"] == "preflight_ready"
    assert saved_state["preflight"]["ok"] is True
    assert saved_state["preflight"]["missing_files"] == []


def test_preflight_does_not_offer_download_without_bowtie(monkeypatch, tmp_path: Path) -> None:
    missing_fasta = tmp_path / "missing.fa"

    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
            "structured_goal": {
                "species": "human",
                "intent": "knockdown",
                "design_programs": [{"modality": "gapmer"}],
            },
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: False)
    monkeypatch.setattr(
        pipeline_tools,
        "_required_tiling_reference_files",
        lambda _species: [str(missing_fasta)],
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("human genome FASTA", [str(missing_fasta)])],
    )
    monkeypatch.setattr(pipeline_tools, "bowtie_1_ready", lambda: False)

    preflight = pipeline_tools._preflight_reference_requirements(str(tmp_path))
    questions, reason, _context = pipeline_tools._reference_preflight_questions(preflight)

    assert preflight["ok"] is False
    assert preflight["downloadable_species"] == []
    assert preflight["missing_requirements"][0].startswith("Bowtie 1")
    assert "download to default" not in questions[0]
    assert "Bowtie 1" in questions[0] or "Bowtie 1" in reason


def test_preflight_fails_when_files_exist_but_bowtie_is_missing(monkeypatch, tmp_path: Path) -> None:
    reference = tmp_path / "human.fa"
    reference.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
            "structured_goal": {"species": "human", "intent": "knockdown", "design_programs": [{"modality": "gapmer"}]},
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: False)
    monkeypatch.setattr(pipeline_tools, "_required_tiling_reference_files", lambda _species: [str(reference)])
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("human genome FASTA", [str(reference)])],
    )
    monkeypatch.setattr(pipeline_tools, "bowtie_1_ready", lambda: False)

    preflight = pipeline_tools._preflight_reference_requirements(str(tmp_path))

    assert preflight["ok"] is False
    assert preflight["missing_files"] == []
    assert preflight["missing_requirements"][0].startswith("Bowtie 1")


def test_reference_download_refuses_without_bowtie(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state: dict[str, Any] = {
        "run_dir": str(run_dir),
        "reference_download_request": {
            "species": ["human"],
            "download_dir": str(tmp_path / "refs"),
        },
    }

    monkeypatch.setattr(
        pipeline_tools,
        "_prepare_execution_stage",
        lambda **_kwargs: {"status": "ok", "state": state, "run_dir": str(run_dir)},
    )
    monkeypatch.setattr(
        pipeline_tools,
        "ensure_bowtie_1",
        lambda **_kwargs: {
            "status": "missing",
            "message": "Bowtie 1 is required",
            "hint": "brew install bowtie",
        },
    )
    monkeypatch.setattr(pipeline_tools, "_save_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_tools, "_mark_last_tool", lambda *_args, **_kwargs: None)

    result = pipeline_tools.tool_download_reference_assets(str(tmp_path / "state.json"))

    assert result["status"] == "fatal_error"
    assert result["message"] == "bowtie_1_not_installed"


def test_reference_provisioning_does_not_reinstall_already_installed_assets(monkeypatch) -> None:
    from test_reference_skip_compare import typed_interpreter
    typed_interpreter(monkeypatch, "continue")
    decision = pipeline_tools._parse_reference_provisioning_decision(
        {
            "clarification": {
                "kind": "reference_preflight",
                "last_answers": [
                    {
                        "question": (
                            "Reference or splice-evaluation assets are missing: "
                            "SpliceTransformer submodule/weights/reference FASTA. "
                            "Should I install them now?"
                        ),
                        "answer": "assets installed; rerun preflight",
                    }
                ]
            }
        },
        {
            "missing_components": ["splice_eval"],
            "missing_requirements": ["SpliceTransformer submodule/weights/reference FASTA"],
            "downloadable_species": [],
        },
    )

    assert decision["action"] == "continue"


def test_preflight_fails_when_files_exist_but_tiling_cli_is_missing(monkeypatch, tmp_path: Path) -> None:
    reference = tmp_path / "human.fa"
    reference.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
            "structured_goal": {"species": "human", "intent": "knockdown", "design_programs": [{"modality": "steric"}]},
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: False)
    monkeypatch.setattr(pipeline_tools, "_required_tiling_reference_files", lambda _species: [str(reference)])
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("human genome FASTA", [str(reference)])],
    )
    monkeypatch.setattr(pipeline_tools, "tiling_cli_tools_ready", lambda: False)
    monkeypatch.setattr(pipeline_tools, "tiling_cli_missing", lambda: ["bedtools", "samtools"])
    monkeypatch.setattr(pipeline_tools, "tiling_cli_install_hint", lambda: "Install BEDTools with Homebrew: brew install bedtools")

    preflight = pipeline_tools._preflight_reference_requirements(str(tmp_path))
    questions, reason, _context = pipeline_tools._reference_preflight_questions(preflight)

    assert preflight["ok"] is False
    assert preflight["missing_files"] == []
    assert "tiling_cli" in preflight["missing_components"]
    assert any(str(item).startswith("Steric tiling CLI") for item in preflight["missing_requirements"])
    assert "Steric tiling CLI" in questions[0]
    assert "install them automatically" in questions[0]
    assert "reference_data_missing" in reason


def test_preflight_fails_when_files_exist_but_snakemake_is_missing(monkeypatch, tmp_path: Path) -> None:
    reference = tmp_path / "human.fa"
    reference.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
            "structured_goal": {"species": "human", "intent": "knockdown", "design_programs": [{"modality": "steric"}]},
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: False)
    monkeypatch.setattr(pipeline_tools, "_required_tiling_reference_files", lambda _species: [str(reference)])
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("human genome FASTA", [str(reference)])],
    )
    monkeypatch.setattr(pipeline_tools, "snakemake_ready", lambda: False)

    preflight = pipeline_tools._preflight_reference_requirements(str(tmp_path))
    questions, reason, _context = pipeline_tools._reference_preflight_questions(preflight)

    assert preflight["ok"] is False
    assert preflight["missing_files"] == []
    assert "Snakemake" in preflight["missing_requirements"]
    assert "snakemake" in preflight["missing_components"]
    assert "Snakemake" in questions[0]
    assert "install them automatically" in questions[0]
    assert "reference_data_missing" in reason


def test_reference_preflight_installs_splice_assets_instead_of_reasking(monkeypatch, tmp_path: Path) -> None:
    from test_reference_skip_compare import typed_interpreter
    typed_interpreter(monkeypatch, "install_splice")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "state" / "run_state.json"
    reference = tmp_path / "human.fa"
    reference.write_text("placeholder", encoding="utf-8")
    checkpoint = tmp_path / "SpTransformer_pytorch.ckpt"
    state = {
        "version": "2",
        "status": "resolution_ready",
        "phase": "resolution_ready",
        "request": "Design human LRRK2 splice-switching ASOs.",
        "flags": {},
        "run_dir": str(run_dir),
        "artifacts": {},
        "events": [],
        "last_error": None,
        "pending_questions": [],
        "clarification": {
            "kind": "reference_preflight",
            "isoform_scope_confirmed": False,
            "last_answers": [
                {
                    "question": "Should I install them now? Answer with 'install splice assets'.",
                    "answer": "install splice assets",
                }
            ],
        },
        "clarification_qa_history": [],
        "force_rerun_stages": [],
        "stage_attempts": {},
    }
    pipeline_tools._save_state(str(state_path), state, snapshot_reason="unit_seed")

    monkeypatch.setattr(
        pipeline_tools,
        "_prepare_execution_stage",
        lambda **_kwargs: {
            "status": "ok",
            "state": state,
            "state_path": str(state_path),
            "run_dir": str(run_dir),
            "force_stage": True,
        },
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_goal_payload_context",
        lambda _run_dir: {
            "structured_goal": {
                "species": "human",
                "intent": "splice_modulation",
                "run_splice_eval": True,
                "design_programs": [{"modality": "steric", "run_splice_eval": True}],
            },
            "final_payload": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "resolve_ensembl_species", lambda _species: {"status": "resolved"})
    monkeypatch.setattr(pipeline_tools, "_resolve_run_splice_eval", lambda _run_dir: True)
    monkeypatch.setattr(pipeline_tools, "resolve_splice_eval_backend", lambda _species: {"backend": "splicetransformer"})
    monkeypatch.setattr(pipeline_tools, "_required_tiling_reference_files", lambda _species: [str(reference)])
    monkeypatch.setattr(
        pipeline_tools,
        "_tiling_reference_requirement_groups",
        lambda _species: [("human genome FASTA", [str(reference)])],
    )

    required = [str(checkpoint)]

    def _required_files() -> list[str]:
        return list(required)

    def _install(**_kwargs):
        checkpoint.write_text("weights", encoding="utf-8")
        return {"status": "ok", "steps": [{"status": "ok", "action": "downloaded"}], "missing_files": []}

    monkeypatch.setattr(pipeline_tools, "_required_human_splice_eval_files", _required_files)
    monkeypatch.setattr(pipeline_tools, "ensure_splicetransformer_assets", _install)

    result = pipeline_tools.tool_reference_preflight(str(state_path))

    assert result["status"] == "ok"
    assert result.get("needs_user_input") is not True
    saved_state = pipeline_tools._load_state(str(state_path))
    assert saved_state.get("preflight", {}).get("ok") is True
