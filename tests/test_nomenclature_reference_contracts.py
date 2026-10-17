import asyncio
import gzip
import io
from copy import deepcopy
from types import SimpleNamespace

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord

from aso_agents import exon_nomenclature as nomenclature, pipeline_tools
import run_goal_intake


def source_fixture(tmp_path, monkeypatch, *, refseq_lengths=(11, 17, 19, 23)):
    annotation = tmp_path / "gencode.v99.annotation.gtf"
    annotation.write_text("# neutral annotation fixture\n")
    with gzip.open(tmp_path / "gencode.v99.metadata.RefSeq.gz", "wt") as handle:
        handle.write("ENST00000000001.4\tNM_999999.2\tNP_999999.1\n")
    record = SeqRecord(Seq("A" * sum(refseq_lengths)), id="NM_999999.2", name="NM_999999", description="neutral test transcript")
    record.annotations.update(molecule_type="mRNA", comment="The four exons are historically designated 1, 2a, 2, and 3.")
    offset = 0
    for length in refseq_lengths:
        record.features.append(SeqFeature(FeatureLocation(offset, offset + length), type="exon"))
        offset += length
    body = io.StringIO()
    SeqIO.write(record, body, "genbank")
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        assert "eutils.ncbi.nlm.nih.gov" in url, "exact local version pair should avoid remote xref dependency"
        assert kwargs["params"]["id"] == "NM_999999.2"
        return SimpleNamespace(text=body.getvalue(), url=url, raise_for_status=lambda: None)
    monkeypatch.setattr(nomenclature.requests, "get", get)
    return annotation, calls


def test_installed_exact_version_pair_and_ordered_structure(tmp_path, monkeypatch):
    annotation, calls = source_fixture(tmp_path, monkeypatch)
    source = nomenclature.fetch_authoritative_exon_nomenclature_source(
        "ENST00000000001.4", expected_exon_count=4, reference_annotation=str(annotation),
        expected_exon_lengths=[11, 17, 19, 23])
    assert source["status"] == "ready"
    assert source["source_ensembl_transcript_id"] == "ENST00000000001.4"
    assert source["refseq_exon_lengths"] == source["ensembl_exon_lengths"]
    assert source["xref_record"]["source"] == "local_annotation_version_pair"
    assert len(source["xref_sha256"]) == 64 and len(calls) == 1


def test_equal_exon_count_does_not_prove_equal_structure(tmp_path, monkeypatch):
    annotation, _ = source_fixture(tmp_path, monkeypatch, refseq_lengths=(11, 18, 18, 23))
    source = nomenclature.fetch_authoritative_exon_nomenclature_source(
        "ENST00000000001.4", expected_exon_count=4, reference_annotation=str(annotation),
        expected_exon_lengths=[11, 17, 19, 23])
    assert source["status"] == "unresolved"
    assert source["fetch_failures"][0]["exception_type"] == "ordered_exon_lengths_mismatch"


def test_local_cross_reference_never_strips_a_conflicting_version(tmp_path, monkeypatch):
    annotation, _ = source_fixture(tmp_path, monkeypatch)
    assert nomenclature._local_annotation_xrefs("ENST00000000001.3", str(annotation)) is None
    assert nomenclature._local_annotation_xrefs("ENST00000000001", str(annotation)) is None


def test_unresolved_mapper_is_not_promoted_to_ordinal_default(tmp_path, monkeypatch):
    annotation, _ = source_fixture(tmp_path, monkeypatch)
    async def unresolved(*_args, **_kwargs):
        return {"status": "unresolved", "reason": "The supplied naming order is ambiguous."}
    monkeypatch.setattr(nomenclature, "_run_schema_agent", unresolved)
    exons = [{"index": i, "start": i * 100, "end": i * 100 + n - 1}
             for i, n in enumerate([11, 17, 19, 23], 1)]
    result = asyncio.run(nomenclature.resolve_transcript_exon_nomenclature(
        transcript_id="ENST00000000001.4", requested_exon_label=2, transcript_exons=exons,
        reference_annotation=str(annotation)))
    assert result["status"] == "unresolved"
    assert "ambiguous" in result["reason"]


def test_biological_label_mapping_is_applied_once():
    mappings = {str(label): {"status": "verified_alternate_mapping", "resolved_transcript_ordinal": label + 1}
                for label in (2, 3)}
    transcript = {"transcript_id": "NEUTRAL_TRANSCRIPT", "chrom": "chr2", "strand": "+",
        "metadata": {"exon_nomenclature_resolutions": mappings},
        "exons_tx": [{"index": i, "start": i * 100, "end": i * 100 + 30} for i in range(1, 6)],
        "sites": [{"exon_index": i, "chrom": "chr2", "strand": "+", "acceptor": i * 100,
                   "donor": i * 100 + 30, "coding": True} for i in range(2, 5)]}
    goal = {"exon_label": 2, "focus_exons_only": True, "desired_effect": "inclusion",
            "splice_isoform_objectives": [{"isoform_label": "neutral", "exon_label": 2,
                                            "desired_abundance": "increase"}]}
    original = deepcopy(goal)
    plan, _ = run_goal_intake._compute_splice_site_plan(goal["splice_isoform_objectives"], {"neutral": transcript}, goal)
    assert {site["position"] for site in plan} == {300, 330}
    assert goal["exon_label"] == original["exon_label"] == 2


def test_ordinal_geometry_cannot_attest_a_named_feature_mapping():
    finding = {"record_index": 0, "status": "satisfied", "requirement_kind": "positive_mapping"}
    record = {"record_index": 0, "resolved_target": {
        "evidence_capabilities": ["ordinal_identity", "exact_reference_identity", "annotated_junction_geometry"]}}
    assert pipeline_tools._resolved_target_finding_capability_errors([finding], [record], positive_status="satisfied")
    record["resolved_target"]["evidence_capabilities"].append("explicit_positive_mapping")
    assert not pipeline_tools._resolved_target_finding_capability_errors([finding], [record], positive_status="satisfied")
