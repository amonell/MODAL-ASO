"""Complete the reference catalog from typed design-program declarations."""
from copy import deepcopy

from .gapmer_selectivity import resolve_catalog_label


def _verified_accession(entry, reference):
    metadata = entry.get('metadata') or {}
    receipts = [metadata.get('reference_accession_resolution'),
                *(metadata.get('reference_accession_resolutions') or [])]
    return any(isinstance(receipt, dict) and receipt.get('exact_accession_verified') is True
               and str(receipt.get('requested_accession') or '').casefold() == reference.casefold()
               for receipt in receipts)


def _declared_structure(runtime, reference, goal):
    """Resolve external accessions before using the Ensembl structure endpoint."""
    physical_id, provenance = reference, {}
    if runtime._looks_like_refseq_transcript_accession(reference):
        from .transcript_chooser_agent import resolve_external_transcript_accession, _get_ensembl_json
        resolution = resolve_external_transcript_accession(reference, species=goal.get("species") or "human") or {}
        physical_id = resolution.get("transcript_id")
        if resolution.get("ambiguous") or not physical_id:
            raise ValueError("declared_target_reference_external_mapping_unresolved: " + reference)
        exact = (resolution.get("requested_accession") == reference
                 and resolution.get("method") == "local_gencode_refseq_xref")
        if not exact:
            _, records = _get_ensembl_json(f"/xrefs/id/{physical_id}?external_db=RefSeq_mRNA")
            exact = any(reference == str(record.get("display_id") or "")
                        for record in records if isinstance(record, dict)) if isinstance(records, list) else False
        if not exact:
            raise ValueError("declared_target_reference_exact_accession_unverified: " + reference)
        provenance = {**resolution, "requested_accession":reference, "exact_accession_verified":True}
    structure = runtime._fetch_transcript_structure(physical_id)
    if not isinstance(structure, dict) or not runtime._source_identifier_matches(structure, physical_id):
        raise ValueError("declared_target_reference_identity_unresolved: " + reference)
    return structure, physical_id, provenance


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
        values = [program.get("transcript_id"), *cohort,
                  (program.get("splice_event") or {}).get("reference_transcript_id")]
        for value in values:
            if isinstance(value, str) and value.strip():
                identifiers = requests.setdefault(value.strip(), [])
                if program.get("program_id") not in identifiers:
                    identifiers.append(program.get("program_id"))
    # The locked accession remains authoritative when a planner emits only its
    # physical Ensembl ID, or a legacy feature request has no program at all.
    # Attest that alias before downstream selectors inspect the existing catalog.
    lock = goal.get("transcript_lock") if goal.get("transcript_policy") == "user_locked" else None
    if isinstance(lock, dict):
        reference = lock.get("value") or lock.get("transcript_id") or lock.get("reference_accession")
        if isinstance(reference, str) and runtime._looks_like_refseq_transcript_accession(reference.strip()):
            requests.setdefault(reference.strip(), [])
    from .preservation_context import preservation_resolution_objectives
    for objective in preservation_resolution_objectives(goal):
        reference = objective.get("transcript_id")
        if isinstance(reference, str) and reference.strip():
            requests.setdefault(reference.strip(), [])
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
        # A human-facing catalog label is not an accession attestation. Intake
        # may already have selected the right physical transcript but retained
        # only a prose match rationale; verify that mapping through the normal
        # resolver before downstream feature selectors rely on it.
        previous_label = label
        needs_attestation = (label is not None and runtime._looks_like_refseq_transcript_accession(reference)
                             and not _verified_accession(catalog[label], reference))
        if label is None or needs_attestation:
            structure, physical_id, provenance = _declared_structure(runtime, reference, goal)
            _check_context(runtime, structure, goal, reference)
            exons = structure.get("exons_tx") or []
            if not exons or not structure.get("seq_region_name") or str(structure.get("strand")) not in {"1", "-1", "+", "-"}:
                raise ValueError("declared_target_reference_structure_incomplete: " + reference)
            if any(not isinstance(exon, dict) or type(exon.get("start")) is not int
                   or type(exon.get("end")) is not int or not 1 <= exon["start"] <= exon["end"]
                   for exon in exons):
                raise ValueError("declared_target_reference_structure_incomplete: " + reference)
            # Do not replace a conflicting existing identity or its provenance.
            if reference in catalog and previous_label is None:
                raise ValueError("declared_target_reference_catalog_conflict: " + reference)
            structure = {**structure, "strand": -1 if str(structure["strand"]) in {"-", "-1"} else 1}
            entry = {
                "transcript_id": physical_id,
                "metadata": {**runtime.resolved_transcript_metadata(structure, {}),
                    **({"reference_accession_resolution":provenance} if provenance else {})},
                "sites": runtime._extract_splice_sites(structure),
                "exons_tx": deepcopy(exons),
                "chrom": structure["seq_region_name"],
                "strand": "-" if str(structure["strand"]) in {"-", "-1"} else "+",
            }
            # Keep existing biological labels when the accession resolves to
            # exactly their attested structure. Preserve every accession
            # receipt; a new alias must not erase an earlier verified one.
            physical_entry = deepcopy(entry)
            physical_entry["metadata"].pop("reference_accession_resolution", None)
            if previous_label is not None:
                existing = resolve_catalog_label(physical_id, {previous_label: catalog[previous_label]},
                                                 {physical_id: physical_entry})
                if existing is None:
                    raise ValueError("declared_target_reference_catalog_conflict: " + reference)
            else:
                existing = resolve_catalog_label(physical_id, catalog, {physical_id: physical_entry})
            if existing is not None and provenance:
                metadata = catalog[existing].setdefault("metadata", {})
                receipts = metadata.setdefault("reference_accession_resolutions", [])
                if provenance not in receipts:
                    receipts.append(provenance)
                if not isinstance(metadata.get("reference_accession_resolution"), dict):
                    metadata["reference_accession_resolution"] = provenance
                elif not metadata["reference_accession_resolution"].get("exact_accession_verified"):
                    # Retain the earlier unresolved record in provenance.
                    earlier = metadata["reference_accession_resolution"]
                    if earlier not in receipts:
                        receipts.append(earlier)
                    metadata["reference_accession_resolution"] = provenance
            else:
                catalog[reference] = entry
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
