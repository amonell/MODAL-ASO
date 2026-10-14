"""Cross-species design can continue with independently resolved references."""
import run_goal_intake as native


def _goal():
    return {"species": "human", "gene_symbol": "GENE", "gene_id": "ENSG1",
            "cross_species_requirement": {"required": True, "species": ["human", "mouse"]}}


def test_homology_outage_uses_exact_independent_species_gene(monkeypatch):
    monkeypatch.setattr(native, "lookup_gene_by_id", lambda gene_id: {
        "gene_id": gene_id, "display_name": "GENE", "species": "homo_sapiens"})
    monkeypatch.setattr(native, "resolve_ortholog_gene", lambda *_: {
        "status": "error", "error": "ortholog_lookup_failed:Ensembl REST unavailable (HTTP 500)"})
    monkeypatch.setattr(native, "lookup_gene_id", lambda symbol, species: {
        "gene_id": "ENSMUSG2", "display_name": "GENE", "ensembl_species": "mus_musculus",
        "assembly": "GRCm39", "resolution_source": "local_gtf_gene_symbol"})
    targets = native._resolve_cross_species_gene_targets(_goal(), ["human", "mouse"])
    assert targets["mouse"]["status"] == "resolved"
    assert targets["mouse"]["gene_id"] == "ENSMUSG2"
    assert targets["mouse"]["homology_status"] == "unverified_service_unavailable"
    assert targets["mouse"]["resolution_source"] == "independent_species_gene_annotation"
    assert "homology_type" not in targets["mouse"]


def test_homology_outage_rejects_ambiguous_or_wrong_species_gene(monkeypatch):
    monkeypatch.setattr(native, "lookup_gene_by_id", lambda gene_id: {
        "gene_id": gene_id, "display_name": "GENE", "species": "homo_sapiens"})
    monkeypatch.setattr(native, "resolve_ortholog_gene", lambda *_: {
        "status": "error", "error": "ortholog_lookup_failed:Ensembl REST unavailable (HTTP 500)"})
    monkeypatch.setattr(native, "lookup_gene_id", lambda symbol, species: {
        "gene_id": "ENSMUSG2", "display_name": "PARALOG", "ensembl_species": "mus_musculus",
        "assembly": "GRCm39"})
    targets = native._resolve_cross_species_gene_targets(_goal(), ["human", "mouse"])
    assert targets["mouse"]["status"] == "error"
    monkeypatch.setattr(native, "lookup_gene_id", lambda symbol, species: {
        "gene_id": "ENSMUSG2", "display_name": "GENE", "ensembl_species": "homo_sapiens",
        "assembly": "GRCm39"})
    targets = native._resolve_cross_species_gene_targets(_goal(), ["human", "mouse"])
    assert targets["mouse"]["status"] == "error"
    monkeypatch.setattr(native, "lookup_gene_id", lambda symbol, species: {
        "gene_id": "ENSMUSG2", "display_name": "GENE", "ensembl_species": "mus_musculus",
        "assembly": "GRCm38"})
    targets = native._resolve_cross_species_gene_targets(_goal(), ["human", "mouse"])
    assert targets["mouse"]["status"] == "error"


def test_exonic_splice_rationale_is_tied_to_physical_site_and_limited():
    from aso_agents.design_delivery import _physical_splice_site_rationale
    row = {"source_modality": "splice_mod", "physical_splice_elements": []}
    site = {"transcript_accession": "TX.3", "exon_ordinal": 7,
            "cdna_start": 401, "cdna_end": 425}
    rationale = _physical_splice_site_rationale(row, site)
    assert "TX.3 exon 7" in rationale and "401–425" in rationale
    assert "remains unmeasured" in rationale and "no site-specific regulatory motif" in rationale
    assert not _physical_splice_site_rationale({"source_modality": "gapmer"}, site)
