from __future__ import annotations

from pathlib import Path

from aso_agents import splice_eval_assets


def test_ensure_splicetransformer_assets_reuses_local_genome_and_skips_existing(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    st_root = repo / "spliceAI_integration" / "SpliceTransformer"
    tasks = st_root / "tasks_annotate_mutations.py"
    tasks.parent.mkdir(parents=True)
    tasks.write_text("print('ok')\n", encoding="utf-8")
    genome = tmp_path / "GRCh38.fasta"
    genome.write_text(">chr1\nACGT\n", encoding="utf-8")
    ckpt = st_root / "model" / "weights" / "SpTransformer_pytorch.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x" * (splice_eval_assets.MIN_CHECKPOINT_BYTES + 1))

    monkeypatch.setattr(splice_eval_assets, "_existing_human_genome_fasta", lambda: genome)

    result = splice_eval_assets.ensure_splicetransformer_assets(repo_root=repo)
    dest = st_root / "data" / "data_package" / "hg38.fa"

    assert result["status"] == "ok"
    assert dest.exists()
    assert dest.resolve() == genome.resolve()
    assert not any(step.get("action") == "download_checkpoint" for step in result["steps"])


def test_ensure_splicetransformer_assets_inits_submodule_when_missing(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    st_root = repo / "spliceAI_integration" / "SpliceTransformer"
    tasks = st_root / "tasks_annotate_mutations.py"
    called = {}

    def _fake_run(cmd, **_kwargs):
        called["cmd"] = cmd
        tasks.parent.mkdir(parents=True, exist_ok=True)
        tasks.write_text("ok\n", encoding="utf-8")
        return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(splice_eval_assets.subprocess, "run", _fake_run)
    monkeypatch.setattr(splice_eval_assets, "_existing_human_genome_fasta", lambda: None)
    monkeypatch.setattr(
        splice_eval_assets,
        "_ensure_hg38_fasta",
        lambda *_args, **_kwargs: {"status": "ok", "action": "downloaded", "path": "hg38.fa"},
    )
    monkeypatch.setattr(
        splice_eval_assets,
        "_ensure_checkpoint",
        lambda *_args, **_kwargs: {"status": "ok", "action": "downloaded", "path": "ckpt"},
    )
    monkeypatch.setattr(splice_eval_assets, "missing_splicetransformer_files", lambda *_args, **_kwargs: [])

    result = splice_eval_assets.ensure_splicetransformer_assets(repo_root=repo)

    assert result["status"] == "ok"
    assert called["cmd"][:4] == ["git", "submodule", "update", "--init"]
    assert tasks.exists()
