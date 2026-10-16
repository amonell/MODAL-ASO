"""Source-linked design observations for custom writers; no selection or rescoring."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .binding_region_reports import annotated_splice_contexts, physical_splice_element_overlaps
from .candidate_presentation import molecule_key
from .reporting_contracts import generated_sequence_fields
from .splice_prediction_records import IDENTITY, METRICS
from .splice_recommendations import finite_number


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _fields(record, prefix=""):
    """JSON pointers retain nested field identity, explicit nulls and arrays."""
    result = {}
    for key, value in record.items():
        pointer = prefix + "/" + str(key).replace("~", "~0").replace("/", "~1")
        if isinstance(value, dict) and value:
            result.update(_fields(value, pointer))
        else:
            result[pointer] = deepcopy(value)
    return result


def _predictions(row):
    records = [("", row)]
    for name in ("source_annotations", "splice_eval_summary"):
        if isinstance(row.get(name), dict):
            records.append(("/" + name, row[name]))
    for prefix, record in list(records):
        sites = record.get("splice_site_deltas")
        if isinstance(sites, list):
            records.extend((f"{prefix}/splice_site_deltas/{i}", site)
                           for i, site in enumerate(sites) if isinstance(site, dict))
    result = []
    for pointer, record in records:
        measured = {key: deepcopy(record[key]) for key in METRICS if key in record}
        if not measured:
            continue
        result.append({
            "source_pointer": pointer,
            "identity": {key: deepcopy(record[key]) for key in IDENTITY if key in record},
            "recorded_values": measured,
            "numeric_values": {key: value for key, raw in measured.items()
                               if (value := finite_number(raw)) is not None},
            "unresolved_fields": [key for key, raw in measured.items()
                                  if finite_number(raw) is None],
        })
    return result


def _program_ids(row, source, goal):
    """Use recorded producer/job associations, never physical region labels."""
    jobs = [job for name in ("steric_design_jobs", "gapmer_design_jobs")
            for job in goal.get(name) or [] if isinstance(job, dict) and job.get("program_id")]
    candidates = []
    if row.get("program_id"):
        candidates.append({str(row["program_id"])})
    if row.get("job_id"):
        candidates.append({str(job["program_id"]) for job in jobs if job.get("job_id") == row["job_id"]})
    origin = {str(job["program_id"]) for job in jobs if job.get("output_dir")
              and Path(source["path"]).resolve().is_relative_to(Path(job["output_dir"]).resolve())}
    if origin:
        candidates.append(origin)
    group = row.get("chemistry_group_id")
    by_group = {str(job["program_id"]) for job in jobs if group and job.get("chemistry_group_id") == group}
    if len(by_group) == 1:
        candidates.append(by_group)
    if not candidates:
        return []
    bound = set.intersection(*candidates)
    return sorted(bound) if len(bound) == 1 else []


def _observation(row, payload, source, index, evidence, goal, intervals_for_row):
    sequences = generated_sequence_fields(row)
    programs = _program_ids(row, source, goal)
    declared = {str(program.get("program_id")): program
                for program in goal.get("design_programs") or [] if isinstance(program, dict)}
    references = sorted({str(declared[program]["transcript_id"]) for program in programs
                         if program in declared and declared[program].get("transcript_id")})
    schema = payload.get("schema") if isinstance(payload, dict) else None
    native = (isinstance(schema, dict) and schema.get("table") == "aso_tiles") or (
        isinstance(payload, dict) and payload.get("artifact_type") == "generated_aso_set")
    blocks = intervals_for_row(row, output_space="genomic") if native else []
    strand = row.get("Strand", row.get("strand"))
    chemistry = molecule_key(row)[1]
    identity = {"task_id": evidence["task_id"], "program_ids": programs,
                "transcript_ids": references, "species": goal.get("species"),
                "genome_build": goal.get("genome_build"),
                "binding_intervals_bed0_half_open": blocks, "strand": strand,
                "target_5to3": sequences.get("target_binding_sequence"),
                "antisense_5to3": sequences.get("synthesized_antisense_sequence"),
                "recorded_chemistry": chemistry}
    # Unspecified chemistry/programs or ambiguous sequence roles cannot establish
    # equivalence. This key identifies a binding, never a score or selected rank.
    verified = (native and blocks and strand in {"+", "-"} and programs
                and all(program in declared for program in programs)
                and identity["target_5to3"] and identity["antisense_5to3"]
                and chemistry != "null")
    catalog = goal.get("isoform_transcripts")
    catalog = {key: value for key, value in catalog.items()
               if isinstance(value, dict) and value.get("strand") == strand} if isinstance(catalog, dict) else {}
    elements = physical_splice_element_overlaps(blocks, catalog)
    for element in elements:
        start, end = element["target_interval_1based_inclusive"]
        anchor = element["boundary_1based"]
        element["relative_interval_inclusive"] = (
            [start - anchor, end - anchor] if strand == "+" else [anchor - end, anchor - start])
    sequence = identity["antisense_5to3"]
    metrics = None
    if sequence and not set(sequence) - set("ACGT"):
        longest = run = 0
        prior = None
        for base in sequence:
            run = run + 1 if base == prior else 1
            longest = max(longest, run)
            prior = base
        metrics = {"length": len(sequence), "gc_bases": sum(x in "GC" for x in sequence),
                   "longest_homopolymer_run": longest, "basis": "verified antisense sequence"}
    provenance = {"path": source["path"], "sha256": source["sha256"],
                  "row_collection": source["row_collection"], "row_index_0based": index}
    return {"observation_id": _digest(provenance), "source": provenance,
            "stage_roles": source["roles"], "binding_identity": identity,
            "binding_match_key": _digest(identity) if verified else None,
            "recorded_fields": _fields(row), "splice_predictions": _predictions(row),
            "derived_sequence_metrics": metrics,
            "physical_splice_elements": elements,
            "annotated_splice_contexts": annotated_splice_contexts(blocks, catalog),
            "geometry_scope": "Recorded transcript catalog and native genomic mapping only; an empty list does not establish absence outside that catalog."}


def write_design_observations(evidence, directory, intervals_for_row):
    """Materialize every recorded row, retaining separate stage observations.

    Matching binding keys permit a scoped join, not replacement of conflicting
    values. Aggregate and job observations retain their original identities.
    """
    root = Path(evidence["producer_run_dir"]).resolve()
    loaded, diagnostics, goal = [], [], {}
    for source in evidence["sources"]:
        path = Path(source["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("design_observation_source_outside_producer")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError("design_observation_source_changed")
        if path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(content)
        except (ValueError, UnicodeError):
            diagnostics.append({"path": str(path), "reason": "unreadable_json"})
            continue
        if "resolved_design_contract" in source["roles"] and isinstance(payload, dict):
            goal = payload
        if not set(source["roles"]).intersection(
                {"raw_generated", "initial_filter", "scored", "final_output", "expression_annotation"}):
            continue
        collection = source["row_collection"]
        rows = payload if collection == "$" else payload.get(collection) if isinstance(payload, dict) else None
        if isinstance(rows, list):
            loaded.append((source, payload, rows))
    observations = []
    for source, payload, rows in loaded:
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                diagnostics.append({"path": source["path"], "row_index_0based": index,
                                    "reason": "non_object_row_retained_in_source"})
                continue
            observations.append(_observation(row, payload, source, index, evidence, goal, intervals_for_row))
    if not observations:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (_digest({"task_id": evidence["task_id"], "producer": str(root)}) + ".json")
    if path.is_symlink() or path.resolve() in {Path(s["path"]).resolve() for s in evidence["sources"]}:
        raise ValueError("design_observation_destination_is_source_or_symlink")
    result = {"schema": "modal-aso-design-observations/1", "task_id": evidence["task_id"],
              "producer_run_dir": str(root), "rows": observations, "diagnostics": diagnostics,
              "reference_sources": [{key: source[key] for key in ("path", "sha256")}
                                    for source in evidence["sources"]
                                    if "resolved_design_contract" in source["roles"]],
              "rules": [
                  "One observation per recorded source row, not one row per unique ASO. Raw aggregate/job sources may overlap.",
                  "Use binding_match_key only within the recorded task/program/reference/chemistry scope. Null keys require explicit source inspection; never join solely by sequence.",
                  "Physical region membership can span several programs. Only explicit producer/job associations establish the program used in binding_identity.",
                  "Keep multiple predictions, precision and conflicts separate by source and site. Missing and nonnumeric values are not zero.",
                  "physical_splice_elements gives annotated exon/intron overlap counts per crossed boundary and source block. It does not predict potency.",
                  "At a donor offset zero is the last exonic base; at an acceptor it is the first exonic base. All relative offsets follow transcript orientation.",
                  "No filtering, ranking or candidate selection is performed by this projection. Preserve the complete requested library and apply the user's criteria separately.",
              ]}
    content = (json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path.write_bytes(content)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
            "roles": ["design_observations"], "row_count": len(observations), "row_collection": "rows",
            "columns": list(observations[0]), "source_count": len(loaded),
            "prediction_records": sum(len(row["splice_predictions"]) for row in observations),
            "diagnostics": diagnostics}
