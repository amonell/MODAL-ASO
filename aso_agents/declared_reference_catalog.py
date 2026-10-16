"""Complete the reference catalog from typed design-program declarations."""
from copy import deepcopy

from .gapmer_selectivity import resolve_catalog_label


def _check_context(runtime, record, goal, reference):
    species = runtime.normalize_species(goal.get("species"))
    actual_species = record.get("species")
    if actual_species and species and runtime.normalize_species(actual_species) != species:
        raise ValueError("declared_target_reference_species_conflict: " + reference)
    actual_gene = record.get("gene_id") or record.get("Parent")
    expected_gene = goal.get("gene_id")
    # Legacy intake accepts a symbol in gene_id until authoritative resolution.
    # Compare that symbol only with the *same reference record's* gene label;
    # the proposed goal's label cannot override a conflicting stable ID.
    expected_symbol = str(expected_gene or "").strip().casefold()
    recorded_symbols = {str(record.get(key) or "").strip().casefold()
                        for key in ("gene_symbol", "gene_name")}
    symbol_matches = bool(expected_symbol and expected_symbol in recorded_symbols)
    if actual_gene and expected_gene and (
            runtime._normalize_ensembl_gene_id(str(actual_gene))
            != runtime._normalize_ensembl_gene_id(str(expected_gene))) and not symbol_matches:
        raise ValueError("declared_target_reference_gene_conflict: " + reference)


def complete_target_references(runtime, goal, final_payload=None):
    """Preserve the anchor while loading every declared local program reference.

    Foreign references declared under cross_species_targets retain their own
    resolver. Neither an accession prefix nor a prose label assigns species.
    An explicit unresolved reference never permits a canonical substitution.
    """
    programs = [p for p in goal.get("design_programs") or [] if isinstance(p, dict)]
    requests = {}
    for program in programs:
        cohort = program.get("transcript_ids") or []
        if not isinstance(cohort, list):
            raise ValueError("declared_target_reference_cohort_must_be_list")
        values = [program.get("transcript_id"), *cohort]
        for value in values:
            if isinstance(value, str) and value.strip():
                identifiers = requests.setdefault(value.strip(), [])
                if program.get("program_id") not in identifiers:
                    identifiers.append(program.get("program_id"))
    for scope in [goal, *programs]:
        for constraint in scope.get("region_constraints") or []:
            if not isinstance(constraint, dict) or constraint.get("source_type") != "transcript_feature":
                continue
            for key in ("reference_transcript_id", "transcript_id"):
                reference = constraint.get(key)
                if isinstance(reference, str) and reference.strip():
                    identifiers = requests.setdefault(reference.strip(), [])
                    if scope.get("program_id") not in identifiers:
                        identifiers.append(scope.get("program_id"))
    if not requests:
        return False
    species = runtime.normalize_species(goal.get("species"))
    foreign = {}
    for name, target in (goal.get("cross_species_targets") or {}).items():
        if not isinstance(target, dict) or runtime.normalize_species(name) == species:
            continue
        for reference in target.get("transcript_ids") or []:
            foreign.setdefault(reference, set()).add(runtime.normalize_species(name))
    catalog = deepcopy(goal.get("isoform_transcripts") or {})
    bindings = []
    for reference, program_ids in requests.items():
        if reference in foreign:
            if len(foreign[reference]) != 1:
                raise ValueError("declared_target_reference_species_ambiguous: " + reference)
            bindings.append({"requested_reference": reference, "program_ids": program_ids,
                             "status": "deferred_to_declared_species",
                             "species": next(iter(foreign[reference]))})
            continue
        label = resolve_catalog_label(reference, catalog, catalog)
        if label is None:
            structure = runtime._fetch_transcript_structure(reference)
            if not isinstance(structure, dict) or not runtime._source_identifier_matches(structure, reference):
                raise ValueError("declared_target_reference_identity_unresolved: " + reference)
            _check_context(runtime, structure, goal, reference)
            exons = structure.get("exons_tx") or []
            if not exons or not structure.get("seq_region_name") or str(structure.get("strand")) not in {"1", "-1", "+", "-"}:
                raise ValueError("declared_target_reference_structure_incomplete: " + reference)
            if any(not isinstance(exon, dict) or type(exon.get("start")) is not int
                   or type(exon.get("end")) is not int or not 1 <= exon["start"] <= exon["end"]
                   for exon in exons):
                raise ValueError("declared_target_reference_structure_incomplete: " + reference)
            # Do not replace a conflicting existing identity or its provenance.
            if reference in catalog:
                raise ValueError("declared_target_reference_catalog_conflict: " + reference)
            structure = {**structure, "strand": -1 if str(structure["strand"]) in {"-", "-1"} else 1}
            catalog[reference] = {
                "transcript_id": reference,
                "metadata": runtime.resolved_transcript_metadata(structure, {}),
                "sites": runtime._extract_splice_sites(structure),
                "exons_tx": deepcopy(exons),
                "chrom": structure["seq_region_name"],
                "strand": "-" if str(structure["strand"]) in {"-", "-1"} else "+",
            }
            label = resolve_catalog_label(reference, catalog, catalog)
            if label is None:
                raise ValueError("declared_target_reference_catalog_ambiguous: " + reference)
        _check_context(runtime, catalog[label], goal, reference)
        _check_context(runtime, catalog[label].get("metadata") or {}, goal, reference)
        bindings.append({"requested_reference": reference, "program_ids": program_ids,
                         "status": "resolved", "catalog_label": label})
    if not any(row["status"] == "resolved" for row in bindings):
        return False
    # Publish atomically only after all local identities have been verified.
    goal["isoform_transcripts"] = catalog
    goal["declared_target_reference_resolution"] = bindings
    if isinstance(final_payload, dict):
        final_payload["isoform_transcripts"] = deepcopy(catalog)
        final_payload["declared_target_reference_resolution"] = deepcopy(bindings)
    return True
