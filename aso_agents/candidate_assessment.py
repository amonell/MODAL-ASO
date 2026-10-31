"""Deterministic, chemistry-aware evidence for candidate selection.

These are computational triage annotations, never safety or potency predictions.
All sequence operations use the attested synthesized antisense strand, 5' to 3'.
"""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from .candidate_presentation import molecule_key

VERSION = "candidate-assessment/2"
IUPAC = {**dict(zip("ACGT", "ACGT")), "R":"AG", "Y":"CT", "S":"GC", "W":"AT",
         "K":"GT", "M":"AC", "B":"CGT", "D":"AGT", "H":"ACT", "V":"ACG", "N":"ACGT"}
MOTIF_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4005641/"
CPG_SOURCE = "https://www.idtdna.com/pages/products/functional-genomics/antisense-oligos"
THERMO_SOURCE = "https://doi.org/10.1021/bi00035a029"


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _positions(sequence, motif):
    motif = motif.upper().replace("U", "T")
    if not motif or set(motif) - IUPAC.keys():
        raise ValueError("Motif must be a nonempty IUPAC sequence, not a regular expression")
    pattern = "".join("[" + IUPAC[base] + "]" for base in motif)
    return [match.start() + 1 for match in re.finditer("(?=" + pattern + ")", sequence)]


def motif_screen(sequence, molecule=None, rules=()):
    """Published context flags plus explicitly requested, configurable exclusions."""
    if not sequence or set(sequence) - set("ACGT"):
        return {"status":"unavailable", "reason":"physical_antisense_sequence_unresolved", "flags":[]}
    molecule = molecule or {}
    sugars = molecule.get("nucleotide_sugars_5to3") or []
    linkages = molecule.get("linkages_5to3") or []
    modifications = molecule.get("base_modifications_5to3") or []
    lna_383 = sugars == ["LNA"] * 3 + ["DNA"] * 8 + ["LNA"] * 3
    cpg_positions = _positions(sequence, "CG")
    linkage_known = len(linkages) == len(sequence) - 1 and all(linkages)
    modifications_known = len(modifications) == len(sequence) and all(modifications)
    full_ps = all(linkage == "PS" for linkage in linkages) if linkage_known else None
    cpg_methylated = (all(modifications[p - 1] == "5mC" for p in cpg_positions)
                      if modifications_known else None)
    if len(sugars) != len(sequence):
        applicability = "chemistry_unresolved"
    elif not lna_383:
        applicability = "outside_studied_architecture"
    elif not linkage_known or not modifications_known:
        applicability = "chemistry_unresolved"
    elif not full_ps or not cpg_methylated or any(value not in {"none", "5mC"} for value in modifications):
        applicability = "outside_studied_chemistry"
    else:
        applicability = "matches_studied_chemistry"
    flags = []
    for motif in ("TGC", "TCC"):
        positions = _positions(sequence, motif)
        if positions:
            flags.append({"rule_id":"lna_383_" + motif.lower(), "motif":motif,
                "positions_1based":positions, "action":"flag", "source":MOTIF_SOURCE,
                "applicability":applicability,
                "study_chemistry":{"lna_383":lna_383, "fully_phosphorothioate":full_ps,
                    "cpg_positions_1based":cpg_positions, "cpg_methylation_matches":cpg_methylated},
                "context":"Association in mouse liver with fully PS LNA 3-8-3 gapmers and methylated CpG motifs; other or unresolved chemistry retains a contextual flag, not a validated toxicity prediction."})
    positions = cpg_positions
    if positions:
        methylated = len(modifications) == len(sequence) and all(modifications[p - 1] == "5mC" for p in positions)
        flags.append({"rule_id":"cpg_immune_context", "motif":"CG", "positions_1based":positions,
            "action":"flag", "source":CPG_SOURCE, "applicability":"context_dependent",
            "cytosines_5mC_at_all_hits":methylated,
            "context":"CpG-associated innate immune recognition depends on chemistry, methylation, sequence and species; a sequence match alone does not predict toxicity."})
    seen = set()
    for rule in rules:
        key = rule["rule_id"]
        if key in seen:
            raise ValueError("Duplicate custom motif rule_id: " + key)
        seen.add(key)
        if rule.get("action") not in {"flag", "exclude"}:
            raise ValueError("Custom motif action must be flag or exclude")
        positions = _positions(sequence, rule["motif"])
        if positions:
            flags.append({"rule_id":key, "motif":rule["motif"], "positions_1based":positions,
                "action":rule["action"], "applicability":"user_defined", "source":"explicit_user_rule",
                "context":rule.get("reason") or "Requested sequence-screening rule; no inferred biological toxicity."})
    return {"status":"screened", "rule_set_version":VERSION, "sequence_orientation":"antisense_5to3",
        "flags":flags, "excluded":any(flag["action"] == "exclude" for flag in flags),
        "interpretation":"Absence of these motif flags is not evidence of safety."}


def duplex_reference(sequence, molecule=None, *, temperature_c=37.0, strand_concentration_nm=500.0):
    """Perfect-match unmodified RNA/DNA NN reference at the parameter set's 1 M Na+.

    No modified-base/sugar/backbone, mismatch, self-structure, or cellular correction
    is invented. The RNA strand is the reverse complement of synthesized ASO DNA.
    """
    from Bio.Seq import Seq
    from Bio.SeqUtils import MeltingTemp as mt
    from Bio import __version__ as biopython_version
    temperature_c, concentration = number(temperature_c), number(strand_concentration_nm)
    if temperature_c is None or temperature_c <= -273.15 or concentration is None or concentration <= 0:
        raise ValueError("Thermodynamic reference requires a physical temperature and positive strand concentration")
    if not sequence or len(sequence) < 2 or set(sequence) - set("ACGT"):
        return {"status":"unavailable", "reason":"unambiguous_physical_sequence_required"}
    rna_dna_alphabet = str(Seq(sequence).reverse_complement())
    complement = str(Seq(rna_dna_alphabet).complement())
    table = mt.R_DNA_NN1
    dh, ds = table["init"]
    for i in range(len(sequence) - 1):
        pair = rna_dna_alphabet[i:i+2] + "/" + complement[i:i+2]
        h, s = table[pair]  # All 16 RNA/DNA oriented Watson-Crick stacks.
        dh, ds = dh + h, ds + s
    tm = mt.Tm_NN(rna_dna_alphabet, nn_table=table, dnac1=concentration,
                  dnac2=concentration, Na=1000, saltcorr=0, selfcomp=False)
    molecule = molecule or {}
    unmodified = (molecule.get("nucleotide_sugars_5to3") == ["DNA"] * len(sequence)
                  and molecule.get("linkages_5to3") == ["PO"] * (len(sequence)-1)
                  and molecule.get("base_modifications_5to3") == ["none"] * len(sequence))
    return {"status":"unmodified_duplex_prediction" if unmodified else "unmodified_reference_proxy",
        "model":"Sugimoto1995_RNA_DNA", "source":THERMO_SOURCE,
        "biopython_version":biopython_version,
        "parameter_table_sha256":hashlib.sha256(json.dumps(table,sort_keys=True).encode()).hexdigest(),
        "complementary_reference_rna_5to3":rna_dna_alphabet.replace("T", "U"),
        "reference_antisense_dna_5to3":sequence, "tm_c":tm,
        "delta_g_kcal_mol":dh - (temperature_c + 273.15) * ds / 1000,
        "delta_h_kcal_mol":dh, "delta_s_cal_mol_k":ds,
        "conditions":{"temperature_c":temperature_c, "sodium_mm":1000,
            "strand_1_nm":concentration, "strand_2_nm":concentration, "salt_correction":"none"},
        "calibration_length_nt":[5,13], "length_extrapolation":not 5 <= len(sequence) <= 13,
        "modified_chemistry_tm_c":None, "modified_chemistry_delta_g_kcal_mol":None,
        "limitation":"Perfect-match unmodified reference at 1 M sodium. Modified ASOs (including 2MOE and PS) have no validated parameters in this calculation. This is not their measured affinity, Kd, potency or specificity; larger Tm is not automatically better."}


def off_target_evidence(row):
    values = {**(row.get("source_annotations") or {}), **row}
    budget = number(values.get("Off_target_mismatch_budget"))
    bins = [number(values.get("Off_target_" + label + "_count")) for label in ("exact","1mm","2mm","3mm")]
    count_status = values.get("off_target_count_status")
    complete_status = (count_status == "complete_within_declared_scope"
                       or (not count_status and values.get("off_target_search_status") == "complete"))
    complete = (complete_status and values.get("off_target_search_status") not in
                {"incomplete_or_censored", "incomplete", "failed", "unavailable", "not_run"}
                and not values.get("off_target_mapping_capped") and budget in {0,1,2,3}
                and all(value is not None and value >= 0 and value.is_integer() for value in bins[:int(budget)+1]))
    return {"status":"complete_within_declared_scope" if complete else "incomplete_or_unavailable",
        "mismatch_budget":int(budget) if budget in {0,1,2,3} else None,
        "counts_by_mismatch":bins, "unit":values.get("off_target_exact_count_unit") or "unresolved",
        "search_status":values.get("off_target_search_status"),
        "references_by_species":values.get("off_target_search_provenance_by_species") or {},
        "species_evidence":values.get("off_target_by_species") or {},
        "species_statuses":values.get("off_target_screening_status_by_species") or {},
        "expression_weighted_risk":number(values.get("Off_target_expression_weighted_risk")),
        "expression_tissue":values.get("Off_target_expression_tissue"),
        "limitation":"Counts are scoped computational matches, not established RNA binding or cellular off-target effects. Missing counts are never zero."}


def assess_candidates(rows, goal):
    result = []
    for original in rows:
        row = deepcopy(original)
        molecule = row.get("molecule_specification") or {}
        sequence = molecule.get("antisense_5to3") or row.get("synthesized_antisense_sequence") or ""
        sequence = sequence.upper().replace("U", "T")
        motifs = motif_screen(sequence, molecule, goal.get("sequence_motif_rules") or [])
        thermo = duplex_reference(sequence, molecule, **(goal.get("thermodynamic_reference_conditions") or {}))
        screen = off_target_evidence(row)
        cryptic = row.get("cryptic_splice_evidence") or (row.get("source_annotations") or {}).get("cryptic_splice_evidence") or {"status":"not_evaluated"}
        if isinstance(cryptic,str):
            try:
                cryptic = json.loads(cryptic)
            except ValueError:
                cryptic = {"status":"unavailable","reason":"malformed_persisted_evidence"}
        missing = []
        if motifs["status"] != "screened":
            missing.append("physical_sequence_motif_screen")
        if screen["status"] != "complete_within_declared_scope":
            missing.append("complete_off_target_screen")
        if not screen["references_by_species"]:
            missing.append("off_target_reference_identity")
        if not molecule:
            missing.append("attested_physical_chemistry")
        accessibility = row.get("target_accessibility")
        if not isinstance(accessibility, dict) or accessibility.get("status") != "measured":
            missing.append("target_RNA_accessibility")
        native_status = row.get("recommendation_status")
        exploratory = native_status in {"opposing_predictions", "no_directional_support",
            "mixed_or_partial_predictions", "incomplete_prediction_evidence"}
        row["candidate_assessment"] = {"version":VERSION, "motifs":motifs,
            "cryptic_splice_evidence":deepcopy(cryptic),
            "duplex_reference":thermo, "off_target":screen, "unavailable_evidence":missing,
            "selection_eligible":not motifs.get("excluded", False) and not (
                motifs["status"] != "screened" and any(rule["action"]=="exclude" for rule in goal.get("sequence_motif_rules") or [])),
            "recommendation_status":"exploratory" if exploratory else "provisional" if missing else "computational_candidate",
            "native_prediction":{"recommendation_tier":row.get("recommendation_tier"),
                "directional_status":row.get("recommendation_status"),
                "selection_rank":row.get("selection_rank"), "ranking_scope":row.get("ranking_scope") or row.get("job_id"),
                "selection_score_components":row.get("selection_score_components")}}
        result.append(row)
    return result


def evidence_sort_key(row):
    """Conservative triage before native within-scope rank, without invented weights.

    Never compare unlike hit units or scopes as a scalar risk score. A completed
    search precedes unavailable evidence; counts are comparable only in a cohort
    with the same search signature (handled by rank_candidates).
    """
    assessment = row.get("candidate_assessment") or {}
    screen = assessment.get("off_target") or {}
    native = row.get("recommendation_rank_key") or []
    tier = number(native[0]) if isinstance(native, (list,tuple)) and native else None
    proximity = number(row.get("requested_exon_proximity_tier")) or 0
    return (not assessment.get("selection_eligible", True), proximity, -(tier if tier is not None else 0),
            screen.get("status") != "complete_within_declared_scope")


def rank_candidates(rows):
    """Preserve native scientific order between incomparable evidence cohorts."""
    groups = {}
    for index, row in enumerate(rows):
        screen = (row.get("candidate_assessment") or {}).get("off_target") or {}
        key = json.dumps([screen.get("mismatch_budget"),screen.get("unit"),screen.get("references_by_species"),
                          screen.get("expression_tissue"),
                          row.get("program_id"),row.get("ranking_scope") or row.get("job_id")], sort_keys=True)
        if screen.get("unit") in {None,"","unresolved"} or not screen.get("references_by_species"):
            key = (key, index)  # Missing search identity cannot establish comparability.
        groups.setdefault(key, []).append((index,row))
    ranked = list(rows)
    for group in groups.values():
        def key(item):
            index,row = item
            assessment = row.get("candidate_assessment") or {}
            screen = assessment.get("off_target") or {}
            counts = (tuple(v if v is not None else math.inf for v in screen.get("counts_by_mismatch", []))
                      if screen.get("status") == "complete_within_declared_scope" else (math.inf,)*4)
            applicable = sum(flag["applicability"] in {"matches_studied_chemistry", "user_defined"}
                             for flag in (assessment.get("motifs") or {}).get("flags", []))
            expression = screen.get("expression_weighted_risk")
            if not screen.get("expression_tissue") or expression is not None and expression < 0:
                expression = None
            return evidence_sort_key(row), counts, (expression if expression is not None else math.inf), applicable, index
        for (slot,_),(_,row) in zip(group, sorted(group,key=key)):
            ranked[slot] = row
    return sorted(ranked, key=evidence_sort_key)


def write_assessment_artifacts(pool, selected, goal, directory):
    """Retain full pre-selection evidence and a distinct, explained shortlist."""
    directory = Path(directory)
    pool_path, shortlist_path = directory / "candidate_assessments.json", directory / "recommended_shortlist.json"
    csv_path, md_path = directory / "recommended_shortlist.csv", directory / "recommended_shortlist.md"
    seen, shortlisted = set(), []
    for row in selected:
        identity = molecule_key(row)
        if identity not in seen:
            seen.add(identity)
            shortlisted.append(row)
    pending = bool(pool) and not shortlist_size_specified(goal)
    if pending:
        shortlisted = []
    payload = {"version":VERSION,"pool_row_count":len(pool),"pool_molecule_count":len({molecule_key(r) for r in pool}),
        "scope":"native_hard_eligible_pool_before_final_panel_selection_and_explicit_motif_exclusions",
        "rows":pool}
    pool_path.write_text(json.dumps(payload, indent=2) + "\n")
    records = []
    for rank, row in enumerate(shortlisted, 1):
        a = row.get("candidate_assessment") or {}
        screen, thermo = a.get("off_target") or {}, a.get("duplex_reference") or {}
        reasons = ["Recorded off-target status: " + str(screen.get("status", "unavailable")) + "."]
        if screen.get("status") == "complete_within_declared_scope":
            reasons.append("Exact through " + str(screen.get("mismatch_budget")) + " substitution counts: " +
                str(screen.get("counts_by_mismatch",[])[:screen["mismatch_budget"]+1]) + " " + str(screen.get("unit")) + ".")
        native = a.get("native_prediction") or {}
        if native.get("directional_status"):
            reasons.append("Native directional evidence: " + str(native["directional_status"]) + ".")
        if a.get("unavailable_evidence"):
            reasons.append("Unavailable evidence: " + ", ".join(a["unavailable_evidence"]) + ".")
        records.append({"recommendation_order":rank,"antisense_5to3":row.get("synthesized_antisense_sequence"),
            "molecule_identity":molecule_key(row),"assessment":a,
            "reason":" ".join(reasons),
            "native_rank":row.get("selection_rank"),"binding_intervals":row.get("binding_intervals_bed0_half_open") or [],
            "off_target_status":screen.get("status"),"off_target_counts":screen.get("counts_by_mismatch"),
            "motif_flags":[flag["rule_id"] for flag in (a.get("motifs") or {}).get("flags",[])],
            "cryptic_splice_evidence":a.get("cryptic_splice_evidence"),
            "reference_tm_c":thermo.get("tm_c"),"reference_delta_g_kcal_mol":thermo.get("delta_g_kcal_mol"),
            "thermodynamic_status":thermo.get("status"),"recommendation_status":a.get("recommendation_status")})
    minimum = goal.get("candidate_minimum")
    shortfall = max(0,minimum-len(records)) if type(minimum) is int else 0
    shortlist = {"version":VERSION,"status":"no_eligible_candidates" if not pool else "awaiting_size_preference" if pending else "insufficient_eligible_candidates" if shortfall else "ready",
        "ranking_use":{"off_target":"Comparable search scopes, count units and tissue evidence only",
            "motifs":"Only applicable or explicit user flags affect ordering; published flags never imply universal toxicity",
            "duplex_reference":"Descriptive reference proxy; never maximized as a potency score",
            "cryptic_splice_evidence":"Review flags with exact genomic scope; no validated biological cutoff is assumed"},
        "selection_policy":"hard_constraints_then_evidence_triage_with_explicit_objective_precedence",
        "pool_artifact":str(pool_path),"pool_sha256":hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "count":len(records),"requested_count":goal.get("candidate_limit"),"minimum_count_shortfall":shortfall,"records":records}
    shortlist_path.write_text(json.dumps(shortlist,indent=2) + "\n")
    fields = ["recommendation_order","antisense_5to3","native_rank","recommendation_status","off_target_status",
              "off_target_counts","motif_flags","cryptic_splice_evidence","reference_tm_c","reference_delta_g_kcal_mol","thermodynamic_status","reason"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in record.items()})
    lines = ["# Recommended shortlist", "", f"{len(records)} distinct molecules selected; {payload['pool_molecule_count']} in the assessed eligible pool.", "",
        "Recommendations combine recorded scientific ranking, comparable off-target screening, and chemistry-qualified motif flags under the requested hard constraints. Explicit joint panel and role objectives retain priority. Unavailable measurements remain visible. Thermodynamic proxies are reported for comparison and are not maximized as a potency score.", "",
        "Tm and ΔG are perfect-match, unmodified RNA/DNA references at 1 M sodium; they are not validated modified-ASO binding affinities. Motif flags are context-dependent and cannot certify safety.", "",
        "Where splice predictions were computed, the CSV/JSON includes the largest unannotated acceptor/donor score gains, genomic coordinates, baseline/perturbed scores and the exact search scope. These are review flags, not confirmed cryptic products; unavailable evidence is not a clean screen.", "",
        "| Order | ASO (5′→3′) | Status | Off-target screen | Motif flags | Reference Tm (°C) | Reference ΔG (kcal/mol) |",
        "|---|---|---|---|---|---|---|"]
    for record in records:
        lines.append("| " + " | ".join(str(record.get(k) if record.get(k) is not None else "unavailable") for k in
            ("recommendation_order","antisense_5to3","recommendation_status","off_target_status","motif_flags","reference_tm_c","reference_delta_g_kcal_mol")) + " |")
    if pending:
        lines.extend(["", "Shortlist size has not been specified; the full assessed pool is preserved while awaiting the user's preference."])
    elif shortfall:
        lines.extend(["", f"The requested minimum is unmet by {shortfall} distinct molecules under the retained constraints."])
    md_path.write_text("\n".join(lines) + "\n")
    return {"candidate_assessments_json":str(pool_path),"recommended_shortlist_json":str(shortlist_path),
            "recommended_shortlist_csv":str(csv_path),"recommended_shortlist_md":str(md_path)}


def shortlist_size_specified(goal):
    if goal.get("shortlist_size_preference") == "all_eligible":
        return True
    def bounded(scope, keys=("candidate_limit", "candidate_limit_per_site", "per_site_candidate_limit")):
        return isinstance(scope, dict) and any(
            type(scope.get(key)) is int and scope[key] > 0 for key in keys)
    if bounded(goal) or goal.get("candidate_role_selection") is not None:
        return True
    programs = goal.get("design_programs") or []
    # A bound on one program says nothing about the size of the others.
    return bool(programs) and all(bounded(program, ("candidate_limit",)) for program in programs)
