"""Evidence-bound gapmer efficacy prediction using the released OligoAI model.

No sequence, chemistry, selection rank or existing score is changed. Predictions
are conditional model estimates, kept separate from measurements and safeguards.
"""
from copy import deepcopy
from functools import lru_cache
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .response_artifacts import digest, write_json
from .oligoai_readiness import OligoAIUnavailable

SOURCE_REVISION = "487e96177e8e2d736654de5b52b39a34401fd0a5"
MODEL_REVISION = "a9d91f4ca370cead8e0571e14a16d48a63d0ecee"
MODEL_SHA256 = "a7f01653d3a2fb3f9dc59e5305642865aa54e06ce029dbb14d7a637a45ce04d9"
MODEL_FILE = "OligoAI_11_09_25.ckpt"
METHODS = {"Electroporation": 0, "Gymnosis": 1, "Other": 2, "Lipofection": 3}
SUGARS = {"DNA": "DNA", "2MOE": "MOE", "cET": "cET"}
SCOPE = ("OligoAI predicts in vitro RNase H-mediated RNA inhibition conditional on "
         "sequence, target RNA context, encoded sugars/bonds, dose and delivery method. "
         "It does not measure efficacy, tissue delivery, toxicity, specificity or clinical response. "
         "Base modifications, PS stereochemistry and cell identity are not model features. "
         "Non-human RNA contexts are extrapolations from the training data. "
         "Predictions are raw, unconstrained percentages; values outside 0–100 are not clipped.")


def reverse_complement(sequence):
    return sequence.upper().replace("U", "T").translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def evaluation_conditions(goal):
    """Use explicit structured conditions, otherwise a labelled reference scenario."""
    options = goal.get("oligoai_evaluation") or (goal.get("constraints") or {}).get("oligoai_evaluation") or {}
    if not isinstance(options, dict):
        raise ValueError("oligoai_evaluation must be an object")
    if set(options) - {"enabled", "dosage_nm", "transfection_method"}:
        raise ValueError("Unknown OligoAI condition field; do not silently discard inputs")
    if "enabled" in options and type(options["enabled"]) is not bool:
        raise ValueError("OligoAI enabled must be boolean")
    dose = options.get("dosage_nm", 100.0)
    method = options.get("transfection_method", "Lipofection")
    if type(dose) not in (int, float) or not math.isfinite(dose) or dose <= 0:
        raise ValueError("OligoAI dosage_nm must be a positive finite number")
    if method not in METHODS:
        raise ValueError("Unknown OligoAI delivery method; no category substitution")
    return {"enabled": options.get("enabled", True), "dosage_nm": float(dose),
            "transfection_method": method,
            "dose_basis": "explicit_structured_input" if "dosage_nm" in options else "assumed_reference_scenario",
            "method_basis": "explicit_structured_input" if "transfection_method" in options else "assumed_reference_scenario"}


def candidate_conditions(row, goal):
    programs = [p for p in goal.get("design_programs") or [] if p.get("program_id") == row.get("program_id") and row.get("program_id")]
    if len(programs) > 1:
        raise ValueError("Ambiguous producing program for OligoAI conditions")
    if not programs or "oligoai_evaluation" not in programs[0]:
        return evaluation_conditions(goal)
    options = goal.get("oligoai_evaluation") or (goal.get("constraints") or {}).get("oligoai_evaluation") or {}
    override = programs[0]["oligoai_evaluation"]
    if not isinstance(options, dict) or not isinstance(override, dict):
        raise ValueError("OligoAI conditions must be objects")
    return evaluation_conditions({"oligoai_evaluation": {**options, **override}})


def _physical_candidate(row):
    """Read retained supplied-table chemistry and confirmed orientation losslessly."""
    candidate = deepcopy(row)
    metadata = row.get("input_metadata") or {}
    for field in ("molecule_specification", "chemistry_idt", "IDT", "chemistry_label", "synthesized_antisense_sequence"):
        if field in metadata and field not in candidate:
            candidate[field] = deepcopy(metadata[field])
    if not candidate.get("chemistry_idt") and metadata.get("ordering_notation"):
        candidate["chemistry_idt"] = metadata["ordering_notation"]
    if row.get("source") == "provided":
        orientation = row.get("orientation_interpretation") or {}
        sequence = orientation.get("evaluated_molecule_sequence")
        if not sequence and row.get("declared_sequence_role") == "antisense_oligo_sequence":
            sequence = row.get("sequence")
        if sequence:
            existing = candidate.get("synthesized_antisense_sequence")
            if existing and existing != sequence:
                raise ValueError("Supplied chemical identity contradicts the evaluated orientation")
            candidate["synthesized_antisense_sequence"] = sequence
    return candidate


def physical_molecule(row, goal):
    from aso_molecule import validate_candidate_molecule
    from .chemical_identity import attest_candidate_rows
    row = _physical_candidate(row)
    if not row.get("molecule_specification"):
        row = attest_candidate_rows([row], {} if row.get("source") == "provided" else goal)[0]
    return validate_candidate_molecule(row)


def model_chemistry(row, goal):
    """Attest the actual molecule, then compile ONLY the released vocabulary."""
    record = physical_molecule(row, goal)
    if record["platform"] != "gapmer":
        raise ValueError("not_gapmer")
    sugars = record["nucleotide_sugars_5to3"]
    if any(value not in SUGARS for value in sugars):
        raise ValueError("unsupported_model_sugar: " + ", ".join(sorted(set(sugars) - set(SUGARS))))
    links = record["linkages_5to3"]
    if any(value not in {"PS", "PO"} for value in links):
        raise ValueError("unsupported_model_backbone")
    if record.get("terminal_modifications") or record.get("conjugates"):
        raise ValueError("unsupported_model_attachment")
    if record["ordering_status"] != "attested":
        raise ValueError("physical_chemistry_not_attested")
    return record, [SUGARS[value] for value in sugars], list(links)


def _is_gapmer(row, goal):
    record = row.get("molecule_specification") or (row.get("input_metadata") or {}).get("molecule_specification") or {}
    try:
        if isinstance(record, str):
            record = json.loads(record)
        if not isinstance(record, dict):
            raise ValueError("Invalid molecule record")
    except (ValueError, TypeError):
        # Include a malformed gapmer record so it receives an explicit error.
        return str(row.get("source_modality") or row.get("platform") or "").lower() == "gapmer"
    if record.get("platform"):
        return record["platform"] == "gapmer"
    if row.get("source_modality"):
        return str(row["source_modality"]).lower() == "gapmer"
    if str(row.get("platform") or "").lower() == "gapmer":
        return True
    # Exact observed notation can establish eligibility without using prose.
    try:
        from .chemical_identity import attest_candidate_rows
        physical = _physical_candidate(row)
        if physical.get("chemistry_idt") or physical.get("IDT"):
            return attest_candidate_rows([physical], {})[0]["molecule_specification"]["platform"] == "gapmer"
    except (ValueError, KeyError, TypeError):
        pass
    from .chemical_identity import candidate_configuration
    try:
        config = candidate_configuration(row, goal)
        return config.get("platform") == "gapmer"
    except (ValueError, TypeError):
        return str(goal.get("intent") or "").lower() in {"knockdown", "gapmer"}


def target_references(goal, runtime, *, catalog=None, cdna=None, genomes=None):
    """Extract from each transcript's bound reference, without wildtype fallback."""
    from .transcript_reference import catalog_genomes
    catalog = deepcopy(catalog if catalog is not None else goal.get("isoform_transcripts") or {})
    species = goal.get("species") or "human"
    patient = runtime._resolve_patient_reference_fasta(goal)
    if cdna is None:
        cdna = runtime._build_isoform_cdna_sequences(catalog, species, reference_fasta=patient)
    if genomes is None:
        genomes = catalog_genomes(catalog, species, runtime, patient)
    references = []
    for label, entry in catalog.items():
        binding = entry.get("reference_binding") or {}
        provenance = {"reference_binding": binding, "sequence_evidence": entry.get("sequence_evidence") or {}}
        common = {"label": str(label), "transcript_id": entry.get("transcript_id") or entry.get("id") or str(label),
                  "species": binding.get("species") or entry.get("species") or species,
                  "assembly": binding.get("assembly"), "provenance": provenance}
        exons = entry.get("exons_tx") or []
        chrom = entry.get("chrom") or entry.get("seq_region_name") or (entry.get("metadata") or {}).get("seq_region_name")
        strand = entry.get("strand")
        if strand in (1, "1"):
            strand = "+"
        elif strand in (-1, "-1"):
            strand = "-"
        genome = (genomes or {}).get(label)
        if exons and chrom and strand in {"+", "-"} and genome is not None:
            start, end = min(int(e["start"]) for e in exons), max(int(e["end"]) for e in exons)
            keys = {str(chrom), "chr" + str(chrom)} if not str(chrom).startswith("chr") else {str(chrom), str(chrom)[3:]}
            present = [key for key in keys if key in genome]
            if str(chrom) in present:
                chrom = str(chrom)
            elif len(present) == 1:
                chrom = present[0]
            else:
                # Retain a separately attested mature-RNA fallback below.
                genome = None
            contig = genome[str(chrom)] if genome is not None else None
            if contig is None:
                if (cdna or {}).get(label):
                    seq = cdna[label].upper().replace("U", "T")
                    references.append({**common, "kind": "mature_rna", "sequence": seq,
                                       "site_start0": 0, "site_end0": len(seq)})
                continue
            lo, hi = max(0, start - 1 - 50), min(len(contig), end + 50)
            sequence = str(contig[lo:hi]).upper()
            if strand == "-":
                sequence = reverse_complement(sequence)
            references.append({**common, "kind": "pre_mrna", "sequence": sequence,
                "chrom": str(chrom), "strand": strand, "genomic_start0": lo, "genomic_end0": hi,
                "site_start0": start - 1 - lo if strand == "+" else hi - end,
                "site_end0": end - lo if strand == "+" else hi - start + 1})
        if (cdna or {}).get(label):
            sequence = cdna[label].upper().replace("U", "T")
            references.append({**common, "kind": "mature_rna", "sequence": sequence,
                               "site_start0": 0, "site_end0": len(sequence)})
    return references


def matching_contexts(sequence, references):
    """Every exact antisense placement, each with 50 RNA-oriented flanking bases."""
    target = reverse_complement(sequence)
    result = []
    premrna_labels = set()
    # Prefer pre-mRNA contexts for a transcript when that reference has a match.
    ordered = sorted(references, key=lambda item: item["kind"] != "pre_mrna")
    for ref in ordered:
        if ref["kind"] == "mature_rna" and ref["label"] in premrna_labels:
            continue
        full = str(ref["sequence"]).upper().replace("U", "T")
        if set(full) - set("ACGTN"):
            raise ValueError("reference contains unsupported nucleotide symbols")
        index = full.find(target)
        while index >= 0:
            end = index + len(target)
            if index >= ref.get("site_start0", 0) and end <= ref.get("site_end0", len(full)):
                lo, hi = max(0, index - 50), min(len(full), end + 50)
                context = "N" * max(0, 50 - index) + full[lo:hi] + "N" * max(0, end + 50 - len(full))
                entry = {key: deepcopy(value) for key, value in ref.items() if key != "sequence"}
                entry.update({"rna_context": context.replace("T", "U"), "start1": index + 1,
                              "end1": end, "reference_sequence_sha256": hashlib.sha256(full.encode()).hexdigest(),
                              "context_basis": "genomic_pre_mrna" if ref["kind"] == "pre_mrna" else "mature_rna_extrapolation",
                              "flank_padding_nt": max(0, 50 - index) + max(0, end + 50 - len(full)),
                              "masked_flank_nt": context.count("N")})
                if ref["kind"] == "pre_mrna":
                    premrna_labels.add(ref["label"])
                    gstart = ref["genomic_start0"] + index if ref["strand"] == "+" else ref["genomic_end0"] - end
                    entry["binding_interval_bed0_half_open"] = [ref["chrom"], gstart, gstart + len(target)]
                result.append(entry)
            index = full.find(target, index + 1)
    return result


def validate_model_input(item):
    if set(item) != {"antisense_5to3", "physical_molecule_sha256", "sugar_mods", "backbone_mods", "rna_context", "dosage_nm", "transfection_method", "input_id"}:
        raise ValueError("Unexpected model input fields")
    molecule_hash = item["physical_molecule_sha256"]
    if not isinstance(molecule_hash, str) or len(molecule_hash) != 64 or set(molecule_hash) - set("0123456789abcdef"):
        raise ValueError("Invalid physical molecule digest")
    sequence = item["antisense_5to3"]
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError("invalid model antisense sequence")
    if len(item["sugar_mods"]) != len(sequence) or any(s not in {"DNA", "MOE", "cET"} for s in item["sugar_mods"]):
        raise ValueError("unsupported or misaligned model sugars")
    if len(item["backbone_mods"]) != len(sequence) - 1 or any(s not in {"PS", "PO"} for s in item["backbone_mods"]):
        raise ValueError("unsupported or misaligned model bonds")
    context = item["rna_context"]
    if len(context) != len(sequence) + 100 or set(context) - set("ACGUN"):
        raise ValueError("invalid model RNA context")
    if context[50:50 + len(sequence)].replace("U", "T") != reverse_complement(sequence):
        raise ValueError("RNA context does not bind the physical antisense molecule")
    if type(item["dosage_nm"]) not in (int, float) or not math.isfinite(item["dosage_nm"]) or item["dosage_nm"] <= 0:
        raise ValueError("invalid model dose")
    if item["transfection_method"] not in METHODS:
        raise ValueError("unknown model delivery method")
    expected = digest({key: value for key, value in item.items() if key != "input_id"})
    if item.get("input_id") != expected:
        raise ValueError("model input digest mismatch")


def runtime_root():
    from .paths import oligoai_root
    return oligoai_root()


@lru_cache(maxsize=4)
def _verified_file(path, size, modified_ns):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _verified_file(str(path), stat.st_size, stat.st_mtime_ns)


def released_model_predict(inputs, directory, *, use_service=True, gpu_lock_fd=None):
    root, directory = runtime_root(), Path(directory)
    python = root / "venv" / "bin" / "python"
    checkpoint, source = root / MODEL_FILE, root / "source"
    identity = {"input_digest": digest(inputs), "runner_sha256": file_sha(Path(__file__).with_name("oligoai_runner.py"))}
    cached = directory / "model-results.json"
    if cached.exists():
        result = json.loads(cached.read_text())
        if all(result.get(k) == v for k, v in identity.items()) and result.get("model_sha256") == MODEL_SHA256 and result.get("source_revision") == SOURCE_REVISION:
            return result
    request = directory / "model-inputs.json"
    write_json(request, {"schema_version": "modal-aso-oligoai-input/1.0", "inputs": inputs})
    from .oligoai_readiness import inspect_installation, OligoAIUnavailable
    readiness = inspect_installation(root)
    if readiness["status"] != "installed":
        raise OligoAIUnavailable(readiness["detail"])
    if file_sha(checkpoint) != MODEL_SHA256:
        raise ValueError("OligoAI released checkpoint checksum mismatch")
    command = [str(python), "-m", "aso_agents.oligoai_runner", "--input", str(request),
               "--output", str(directory / "model-results.json"), "--root", str(root)]
    env = dict(os.environ)
    env.pop("ASO_OLIGOAI_GPU_LOCK_FD", None)
    if gpu_lock_fd is not None:
        env["ASO_OLIGOAI_GPU_LOCK_FD"] = str(gpu_lock_fd)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    env["TMPDIR"] = str(directory / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    with (directory / "stdout.txt").open("w") as stdout, (directory / "stderr.txt").open("w") as stderr:
        subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=True,
                       pass_fds=(gpu_lock_fd,) if gpu_lock_fd is not None else ())
    result = json.loads((directory / "model-results.json").read_text())
    if (result.get("model_sha256") != MODEL_SHA256 or result.get("source_revision") != SOURCE_REVISION
            or any(result.get(k) != v for k, v in identity.items())):
        raise ValueError("OligoAI inference returned an unexpected model identity")
    return result


def evaluate_candidates(rows, goal, output_dir, *, references=None, runtime=None, predictor=None):
    """Evaluate all supported gapmers; carry per-row noncompletion explicitly."""
    candidates = [(i, row) for i, row in enumerate(rows, 1) if _is_gapmer(row, goal)]
    if not candidates:
        return {}
    result = {"schema_version": "modal-aso-oligoai-evaluation/1.0", "scope": SCOPE,
              "model_sha256": MODEL_SHA256, "source_revision": SOURCE_REVISION,
              "model_revision": MODEL_REVISION, "candidates": []}
    try:
        conditions = evaluation_conditions(goal)
        result["conditions"] = conditions
    except (ValueError, TypeError) as exc:
        conditions = None
        result["error"] = str(exc)
    needs_references = False
    for _, row in candidates:
        try:
            needs_references |= candidate_conditions(row, goal)["enabled"]
        except (TypeError, ValueError):
            pass
    if needs_references:
        try:
            references = references if references is not None else target_references(goal, runtime)
        except Exception as exc:
            references = []
            result["reference_error"] = f"{type(exc).__name__}: {exc}"
    else:
        references = []
    inputs = {}
    for index, row in candidates:
        identity = row.get("input_row_id") or f"candidate_{index}"
        entry = {"candidate_id": identity, "candidate_number": index, "name": row.get("name") or identity,
                 "status": "unavailable", "contexts": []}
        result["candidates"].append(entry)
        try:
            row_conditions = candidate_conditions(row, goal)
            entry["conditions"] = row_conditions
        except (ValueError, TypeError) as exc:
            entry["reason"] = str(exc)
            continue
        if not row_conditions["enabled"]:
            entry.update(status="disabled", reason="explicitly_disabled")
            continue
        try:
            record = physical_molecule(row, goal)
            entry.update(antisense_5to3=record["antisense_5to3"], physical_molecule_sha256=record["physical_molecule_sha256"])
            entry["unencoded_chemistry"] = {key: record[key] for key in ("base_modifications_5to3", "PS_stereochemistry")}
            record, sugars, bonds = model_chemistry(row, goal)
            contexts = matching_contexts(record["antisense_5to3"], references)
            if not contexts:
                raise ValueError("no_exact_antisense_site_in_available_target_references")
            for context in contexts:
                item = {"antisense_5to3": record["antisense_5to3"], "physical_molecule_sha256": record["physical_molecule_sha256"],
                        "sugar_mods": sugars, "backbone_mods": bonds, "rna_context": context["rna_context"],
                        "dosage_nm": row_conditions["dosage_nm"], "transfection_method": row_conditions["transfection_method"]}
                item["input_id"] = digest(item)
                validate_model_input(item)
                inputs[item["input_id"]] = item
                entry["contexts"].append({**context, "input_id": item["input_id"]})
            entry["status"] = "pending"
        except (ValueError, KeyError, TypeError) as exc:
            entry["reason"] = str(exc)
            if str(exc).startswith("unsupported_model_"):
                entry["status"] = "unsupported"
    directory = Path(output_dir) / "oligoai" / digest({"evaluation": result, "inputs": list(inputs.values())})
    directory.mkdir(parents=True, exist_ok=True)
    if inputs:
        try:
            model = (predictor or released_model_predict)(list(inputs.values()), directory)
            values = model["predictions"]
            ids = [v["input_id"] for v in values]
            if len(ids) != len(set(ids)) or set(ids) != set(inputs):
                raise ValueError("OligoAI output identity/coverage mismatch")
            by_id = {v["input_id"]: v["predicted_inhibition_percent"] for v in values}
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in by_id.values()):
                raise ValueError("OligoAI returned a nonfinite/non-numeric prediction")
            result["runtime"] = {k: v for k, v in model.items() if k != "predictions"}
            for entry in result["candidates"]:
                if entry["status"] == "pending":
                    entry["status"] = "predicted"
                    for context in entry["contexts"]:
                        context["predicted_inhibition_percent"] = by_id[context["input_id"]]
        except OligoAIUnavailable as exc:
            result["runtime_availability"] = {"status": "unavailable", "detail": str(exc)}
            for entry in result["candidates"]:
                if entry["status"] == "pending":
                    entry.update(status="unavailable", reason=str(exc))
        except Exception as exc:
            result["inference_error"] = f"{type(exc).__name__}: {exc}"
            for entry in result["candidates"]:
                if entry["status"] == "pending":
                    entry.update(status="error", reason=result["inference_error"])
    result["status"] = "complete" if all(c["status"] == "predicted" for c in result["candidates"]) else "incomplete"
    result["predicted_count"] = sum(c["status"] == "predicted" for c in result["candidates"])
    result["candidate_count"] = len(candidates)
    result["artifact_json"] = str(directory / "evaluation.json")
    result["artifact_csv"] = str(directory / "predictions.csv")
    write_json(result["artifact_json"], result)
    fields = ["candidate_id", "antisense_5to3", "status", "reason", "transcript_id", "species", "context_basis", "start1", "end1", "predicted_inhibition_percent", "dosage_nm", "transfection_method", "dose_basis", "method_basis", "physical_molecule_sha256", "input_id"]
    with Path(result["artifact_csv"]).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in result["candidates"]:
            for context in entry["contexts"] or [{}]:
                writer.writerow({**entry, **context, **entry.get("conditions", {})})
    for index, entry in enumerate(result["candidates"]):
        candidates[index][1]["oligoai_evaluation"] = deepcopy(entry)
    return result


def markdown_section(evaluation, *, report_dir=None):
    if not evaluation or not evaluation.get("candidates"):
        return ""
    cond = evaluation.get("conditions") or {}
    lines = ["## OligoAI gapmer efficacy prediction", "", SCOPE, ""]
    def basis(value):
        return "specified" if value == "explicit_structured_input" else "assumed reference scenario"
    if cond:
        lines.append(f"Conditions: **{cond['dosage_nm']:g} nM**, **{cond['transfection_method']}**. "
                     f"Dose: {basis(cond['dose_basis'])}; delivery: {basis(cond['method_basis'])}. "
                     "Per-candidate conditions below take precedence where specified.")
    lines.extend(["", "Mature-RNA contexts are labelled as extrapolations from the genomic pre-mRNA context convention. "
                  "Unencoded base modifications and stereochemistry remain in the molecule records; no chemical substitution was made.", "",
                  "| Candidate | ASO (5′→3′) | Reference / context | Dose (nM) / delivery | Predicted inhibition (%) | Status |",
                  "|---|---|---|---|---:|---|"])
    def cell(v):
        return html.escape(str(v or "")).replace("|", "&#124;").replace("\n", " ")
    for entry in evaluation["candidates"]:
        for context in entry["contexts"] or [{}]:
            prediction = context.get("predicted_inhibition_percent")
            value = f"{prediction:.2f}" if prediction is not None else "—"
            interval = context.get("binding_interval_bed0_half_open")
            placement = f"{interval[0]}:{interval[1]+1}–{interval[2]} (genomic)" if interval else f"cDNA {context.get('start1', '')}–{context.get('end1', '')}"
            reference = (f"{context.get('species', '')} {context.get('transcript_id', '')} "
                         f"{context.get('context_basis', '')} {placement}") if context else "—"
            if context.get("masked_flank_nt"):
                reference += f"; {context['masked_flank_nt']} unavailable flanking bases masked"
            row_cond = entry.get("conditions") or {}
            condition_text = (f"{row_cond['dosage_nm']:g} / {row_cond['transfection_method']} "
                              f"(dose {basis(row_cond['dose_basis'])}; delivery {basis(row_cond['method_basis'])})") if row_cond else "—"
            lines.append("| " + " | ".join([cell(entry['candidate_id']), cell(entry.get('antisense_5to3')), cell(reference),
                                           cell(condition_text), value, cell(entry['status'] + (": " + entry['reason'] if entry.get('reason') else ""))]) + " |")
    lines.extend(["", f"Model: `{MODEL_FILE}`; OligoAI source `{SOURCE_REVISION[:12]}`. "
                  f"Evaluated {evaluation.get('predicted_count', 0)}/{evaluation.get('candidate_count', 0)} gapmers."])
    for label, field in (("Complete OligoAI evidence", "artifact_json"), ("OligoAI predictions CSV", "artifact_csv")):
        if evaluation.get(field):
            path = str(evaluation[field])
            if report_dir:
                path = os.path.relpath(path, str(report_dir))
            lines.append(f"- [{label}](<{path}>)")
    return "\n".join(lines) + "\n"


def html_section(evaluation, *, report_dir=None):
    if not evaluation or not evaluation.get("candidates"):
        return ""
    # Static HTML is visible without JavaScript and keeps the native report style.
    lines = markdown_section(evaluation, report_dir=report_dir).splitlines()
    parts, table = [], []
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("| "):
            values = line[2:-2].split(" | ")
            tag = "th" if not table else "td"
            table.append("<tr>" + "".join(f"<{tag}>{value}</{tag}>" for value in values) + "</tr>")
            continue
        if table:
            parts.append('<div style="overflow-x:auto"><table>' + "".join(table) + "</table></div>")
            table = []
        if line.startswith("## "):
            parts.append("<h2>" + html.escape(line[3:]) + "</h2>")
        elif line.startswith("- ["):
            label, path = line[3:].split("](", 1)
            parts.append('<p><a href="' + html.escape(path[1:-2], quote=True) + '">' + html.escape(label) + '</a></p>')
        elif line:
            parts.append("<p>" + html.escape(line.replace("**", "").replace("`", "")) + "</p>")
    if table:
        parts.append("<table>" + "".join(table) + "</table>")
    return '<section class="panel oligoai-evaluation" id="oligoai-evaluation">' + "".join(parts) + '</section>'


def report_evaluations(payload):
    """Find distinct persisted evaluations without performing inference."""
    evaluations, seen = [], set()
    def visit(value):
        if isinstance(value, dict):
            if value.get("schema_version") == "modal-aso-oligoai-evaluation/1.0":
                key = digest(value)
                if key not in seen:
                    seen.add(key)
                    evaluations.append(value)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(payload)
    return evaluations


def append_report_evaluations(markdown, payload, *, report_dir=None):
    """Project computed evidence across native, supplied and composite reports."""
    # Native child reports already carry their exact section into a composite.
    for evaluation in report_evaluations(payload):
        section = markdown_section(evaluation, report_dir=report_dir).strip()
        if section and section not in markdown:
            markdown = markdown.rstrip() + "\n\n" + section + "\n"
    return markdown
