import json
from pathlib import Path
import shutil
from urllib.parse import unquote

import pytest

from aso_agents import pipeline_tools as p
from aso_agents.evaluation_report import package_evaluation_evidence, write_evaluation_evidence


def example_payload(name="Observation A"):
    return {"answer_summary": "Activity is unmeasured.",
            "evaluation_summary": {"provided_asos": [{"name": name, "sequence": "AACCGGTT"}]},
            "evaluation_assessments": [{"name": name, "uncertainty": "Activity is unmeasured."}]}


def destinations(markdown):
    # Renderer-owned links in these fixtures; no application parser is reused.
    return [unquote(part.split(")", 1)[0].split("#", 1)[0])
            for part in markdown.split("](")[1:]]


def test_requested_report_is_portable_and_science_unchanged(tmp_path):
    payload = example_payload()
    markdown = "Activity is unmeasured. [row](evaluation_evidence.html#row-Observation%20A)\n[raw](evaluation_evidence.json)"
    canonical = tmp_path / "native"
    export = tmp_path / "exports" / "my evaluation.md"
    result = p._write_v2_response_outputs(run_dir=str(canonical), markdown_text=markdown,
        response_payload=payload, response_basename="existing_aso_evaluation", title="Evaluation",
        requested_outputs=[{"path": str(export), "kind": "rationale"}])
    assert Path(result["report_md"]).read_text() == markdown
    exported = export.read_text()
    assert exported.startswith("Activity is unmeasured.")
    assert "#row-Observation%20A)" in exported
    assert "%20" in exported  # The output filename includes a space.
    assets = [export.parent / relative for relative in destinations(exported)]
    assert all(path.is_file() for path in assets)
    for path in assets:
        assert path.read_bytes() == (canonical / path.name).read_bytes()
    assert json.loads(assets[1].read_text())["provided_asos"] == payload["evaluation_summary"]["provided_asos"]
    portable = tmp_path / "shared"
    shutil.copytree(export.parent, portable)
    shutil.rmtree(canonical)
    shutil.rmtree(export.parent)
    assert all((portable / relative).is_file() for relative in destinations((portable / export.name).read_text()))


def test_distinct_reports_in_one_directory_preserve_prior_evidence(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    first = write_evaluation_evidence(source, example_payload("First"))
    text = "[data](evaluation_evidence.json) [observation](evaluation_evidence.html#row-First)"
    destination = tmp_path / "out" / "report.md"
    old_markdown, old_assets = package_evaluation_evidence(text, destination, first)
    old_bytes = {key: Path(path).read_bytes() for key, path in old_assets.items()}
    second = write_evaluation_evidence(source, example_payload("Second"))
    new_markdown, new_assets = package_evaluation_evidence(text, destination, second)
    assert set(old_assets.values()).isdisjoint(new_assets.values())
    assert old_markdown != new_markdown
    assert all(Path(path).read_bytes() == old_bytes[key] for key, path in old_assets.items())
    assert package_evaluation_evidence(text, destination, second) == (new_markdown, new_assets)


def test_corrupted_existing_bundle_is_not_silently_overwritten(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    evidence = write_evaluation_evidence(source, example_payload())
    destination = tmp_path / "out" / "report.md"
    _, assets = package_evaluation_evidence("[raw](evaluation_evidence.json)", destination, evidence)
    damaged = Path(assets["evaluation_evidence.json"])
    damaged.write_text("corrupt")
    with pytest.raises(ValueError, match="export_evidence_integrity_error"):
        package_evaluation_evidence("[raw](evaluation_evidence.json)", destination, evidence)
    assert damaged.read_text() == "corrupt"


def test_unrelated_prose_and_links_are_unchanged(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    evidence = write_evaluation_evidence(source, example_payload())
    text = "File evaluation_evidence.html retains evidence. [external](https://example.org/evaluation_evidence.html)"
    exported, _ = package_evaluation_evidence(text, tmp_path / "report.md", evidence)
    assert exported == text
