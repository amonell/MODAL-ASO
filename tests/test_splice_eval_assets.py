from pathlib import Path

from aso_agents import splice_eval_assets as assets


def test_model_uses_managed_storage_without_a_source_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_ASO_DATA_DIR", str(tmp_path))
    root = assets.splicetransformer_root()
    assert root == tmp_path / "models/SpliceTransformer"
    assert all(path.is_relative_to(root) for path in assets.required_splicetransformer_files())


def test_pinned_model_reuses_reference_without_copying_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_ASO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODAL_ASO_CACHE_DIR", str(tmp_path / "cache"))
    genome = tmp_path / "references" / assets.CATALOG["human"]["fasta"]
    genome.parent.mkdir()
    genome.write_text(">chr1\nACGT\n")
    Path(str(genome) + ".fai").write_text("chr1\t4\t6\t4\t5\n")
    calls = []
    def source(url, revision, root):
        calls.append(revision)
        root.mkdir(parents=True, exist_ok=True)
        (root / "tasks_annotate_mutations.py").write_text("")
        return {"status": "ready"}
    monkeypatch.setattr(assets, "install_source", source)
    monkeypatch.setattr(assets, "_ensure_checkpoint", lambda *args: {"status": "ready"})
    assert assets.ensure_splicetransformer_assets()["status"] == "ok"
    assert assets.ensure_splicetransformer_assets()["status"] == "ok"
    target = assets.required_splicetransformer_files()[-1]
    assert target.is_symlink() and target.resolve() == genome.resolve()
    assert all(value == assets.SPLICETRANSFORMER_REVISION for value in calls)


def test_model_install_rejects_a_conflicting_genome(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_ASO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODAL_ASO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(assets, "install_source", lambda *args: {})
    monkeypatch.setattr(assets, "_ensure_checkpoint", lambda *args: {})
    genome = tmp_path / "references" / assets.CATALOG["human"]["fasta"]
    genome.parent.mkdir()
    genome.write_text(">chr1\nACGT\n")
    destination = assets.required_splicetransformer_files()[-1]
    destination.parent.mkdir(parents=True)
    destination.write_text(">chr1\nTTTT\n")
    result = assets.ensure_splicetransformer_assets()
    assert result["status"] == "error"
    assert "different genome" in result["error"]
    assert "TTTT" in destination.read_text()
