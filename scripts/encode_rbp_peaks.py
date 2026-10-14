#!/usr/bin/env python3
"""Query ENCODE eCLIP experiments for an RBP and pick the best peak dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_ENCODE = os.environ.get("ENCODE_BASE", "https://www.encodeproject.org")
USER_AGENT = os.environ.get("HTTP_USER_AGENT", "aso-encode/0.1 (+https://example.org)")


def _encode_get_json(
    path_or_url: str,
    params: Optional[Dict[str, str]] = None,
    encode_base: str = DEFAULT_ENCODE,
) -> Dict[str, Any]:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        url = path_or_url
    else:
        url = f"{encode_base.rstrip('/')}/{path_or_url.lstrip('/')}"
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 404 and path_or_url.rstrip("/") == "/search":
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("@graph") == [] and payload.get("total") == 0:
            return payload
    resp.raise_for_status()
    return resp.json()


def _search_experiments(
    rbp: str,
    assay_title: Optional[str],
    include_non_released: bool,
    encode_base: str,
) -> List[Dict[str, Any]]:
    params = {
        "type": "Experiment",
        "limit": "all",
        "format": "json",
        "frame": "object",
        "target.label": rbp,
    }
    if assay_title:
        params["assay_title"] = assay_title
    if not include_non_released:
        params["status"] = "released"
    payload = _encode_get_json("/search/", params=params, encode_base=encode_base)
    graph = payload.get("@graph", [])
    if graph:
        return graph

    params.pop("target.label", None)
    params["searchTerm"] = rbp
    payload = _encode_get_json("/search/", params=params, encode_base=encode_base)
    return payload.get("@graph", [])


def _ensure_experiment_detail(exp: Dict[str, Any], encode_base: str) -> Dict[str, Any]:
    if "files" in exp and exp.get("files"):
        if isinstance(exp["files"][0], dict):
            return exp
    exp_id = exp.get("@id") or exp.get("accession")
    if not exp_id:
        return exp
    return _encode_get_json(exp_id, encode_base=encode_base)


def _iter_file_objects(
    file_refs: Iterable[Any],
    encode_base: str,
) -> Iterable[Dict[str, Any]]:
    for ref in file_refs:
        if isinstance(ref, dict):
            yield ref
        elif isinstance(ref, str):
            yield _encode_get_json(ref, encode_base=encode_base)


def _normalize(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _normalize_assembly(value: Optional[str]) -> str:
    norm = _normalize(value)
    if norm in {"grch38", "hg38"}:
        return "grch38"
    if norm in {"grch37", "hg19"}:
        return "grch37"
    return norm


def _extract_target_label(exp: Dict[str, Any]) -> Optional[str]:
    target = exp.get("target")
    if isinstance(target, dict):
        return target.get("label") or target.get("name")
    return None


def _extract_organism(exp: Dict[str, Any]) -> Optional[str]:
    biosample = exp.get("biosample_ontology") or {}
    organism = biosample.get("organism") or {}
    if isinstance(organism, dict):
        name = organism.get("scientific_name")
        if name:
            return name
    target_organism = (exp.get("target") or {}).get("organism") if isinstance(exp.get("target"), dict) else None
    if isinstance(target_organism, dict) and target_organism.get("scientific_name"):
        return target_organism["scientific_name"]
    if isinstance(exp.get("organism"), dict):
        return exp["organism"].get("scientific_name")
    if isinstance(exp.get("organism"), str):
        return exp["organism"]
    return None


def _extract_biosample(exp: Dict[str, Any]) -> Optional[str]:
    biosample = exp.get("biosample_ontology") or {}
    if isinstance(biosample, dict):
        return biosample.get("term_name") or biosample.get("classification")
    return None


def _looks_like_peak(file_obj: Dict[str, Any]) -> bool:
    output_type = _normalize(file_obj.get("output_type"))
    if "peak" not in output_type:
        return False
    file_format = _normalize(file_obj.get("file_format"))
    file_type = _normalize(file_obj.get("file_type"))
    file_format_type = _normalize(file_obj.get("file_format_type"))
    if file_format == "bed":
        return True
    if "bed" in file_type:
        return True
    if file_format_type in {"narrowpeak", "broadpeak"}:
        return True
    return False


def _score_candidate(
    file_obj: Dict[str, Any],
    exp: Dict[str, Any],
    rbp: str,
    organism: Optional[str],
    assembly: Optional[str],
    biosample_hint: Optional[str],
    include_non_released: bool,
) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []

    output_type = _normalize(file_obj.get("output_type"))
    if output_type == "idr thresholded peaks":
        score += 60
        reasons.append("output_type:IDR thresholded peaks")
    elif "replicated" in output_type and "peaks" in output_type:
        score += 45
        reasons.append("output_type:replicated peaks")
    elif "peaks" in output_type:
        score += 30
        reasons.append("output_type:peaks")

    if _normalize(file_obj.get("file_format_type")) == "narrowpeak":
        score += 12
        reasons.append("file_format_type:narrowPeak")
    elif _normalize(file_obj.get("file_format_type")) == "broadpeak":
        score += 8
        reasons.append("file_format_type:broadPeak")

    if file_obj.get("preferred_default"):
        score += 8
        reasons.append("preferred_default")

    if file_obj.get("status") == "released":
        score += 10
        reasons.append("file_status:released")
    elif not include_non_released and file_obj.get("status"):
        score -= 20

    exp_status = exp.get("status")
    if exp_status == "released":
        score += 8
        reasons.append("experiment_status:released")
    elif not include_non_released and exp_status:
        score -= 15

    target_label = _extract_target_label(exp)
    if target_label and _normalize(target_label) == _normalize(rbp):
        score += 8
        reasons.append("target_label:exact")

    if assembly:
        file_assembly = _normalize_assembly(file_obj.get("assembly"))
        requested = _normalize_assembly(assembly)
        if file_assembly and file_assembly == requested:
            score += 12
            reasons.append(f"assembly:{assembly}")
        else:
            score -= 12

    if organism:
        exp_organism = _extract_organism(exp)
        if exp_organism and _normalize(exp_organism) == _normalize(organism):
            score += 6
            reasons.append(f"organism:{organism}")
        else:
            score -= 6

    if biosample_hint:
        biosample = _extract_biosample(exp)
        if biosample and _normalize(biosample_hint) in _normalize(biosample):
            score += 5
            reasons.append("biosample_hint")

    bio_reps = file_obj.get("biological_replicates")
    if isinstance(bio_reps, list) and len(bio_reps) > 1:
        score += 4
        reasons.append("biological_replicates>1")

    return score, reasons


def _collect_candidates(
    experiments: List[Dict[str, Any]],
    rbp: str,
    organism: Optional[str],
    assembly: Optional[str],
    biosample_hint: Optional[str],
    include_non_released: bool,
    encode_base: str,
    strict_assembly: bool,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for exp in experiments:
        detail = _ensure_experiment_detail(exp, encode_base)
        if not include_non_released and detail.get("status") not in {None, "released"}:
            continue
        target = detail.get("target")
        if isinstance(target, str):
            detail = {**detail, "target": _encode_get_json(target, encode_base=encode_base)}
        if _normalize(_extract_target_label(detail)) != _normalize(rbp):
            continue
        actual_organism = _extract_organism(detail)
        if organism and actual_organism and _normalize(actual_organism) != _normalize(organism):
            continue
        files = detail.get("files", [])
        if not files:
            continue
        for file_obj in _iter_file_objects(files, encode_base):
            if not include_non_released and file_obj.get("status") not in {None, "released"}:
                continue
            if not _looks_like_peak(file_obj):
                continue
            if strict_assembly and assembly:
                file_assembly = _normalize_assembly(file_obj.get("assembly"))
                requested = _normalize_assembly(assembly)
                if file_assembly != requested:
                    continue
            score, reasons = _score_candidate(
                file_obj,
                detail,
                rbp=rbp,
                organism=organism,
                assembly=assembly,
                biosample_hint=biosample_hint,
                include_non_released=include_non_released,
            )
            exp_accession = detail.get("accession")
            file_accession = file_obj.get("accession")
            href = file_obj.get("href")
            download_url = (
                f"{encode_base.rstrip('/')}/{href.lstrip('/')}"
                if href
                else None
            )
            candidates.append(
                {
                    "score": score,
                    "score_breakdown": reasons,
                    "experiment_accession": exp_accession,
                    "experiment_id": detail.get("@id"),
                    "experiment_status": detail.get("status"),
                    "assay_title": detail.get("assay_title"),
                    "target_label": _extract_target_label(detail) or rbp,
                    "biosample_term_name": _extract_biosample(detail),
                    "organism": _extract_organism(detail),
                    "file_accession": file_accession,
                    "file_id": file_obj.get("@id"),
                    "file_status": file_obj.get("status"),
                    "file_format": file_obj.get("file_format"),
                    "file_format_type": file_obj.get("file_format_type"),
                    "output_type": file_obj.get("output_type"),
                    "assembly": file_obj.get("assembly"),
                    "href": href,
                    "download_url": download_url,
                }
            )
    return candidates


def _vocabulary_tokens(value):
    # Match phrases against provider-returned labels, not a hand-maintained
    # list of aliases or biological intent inferred from request wording.
    return tuple(''.join(ch.casefold() if ch.isalnum() else ' ' for ch in str(value)).split())


def resolve_assay_title(requested, available):
    exact = [title for title in available if _vocabulary_tokens(title) == _vocabulary_tokens(requested)]
    if len(exact) == 1:
        return exact[0]
    words = _vocabulary_tokens(requested)
    matches = []
    for title in available:
        phrase = _vocabulary_tokens(title)
        if phrase and any(words[i:i+len(phrase)] == phrase for i in range(len(words)-len(phrase)+1)):
            matches.append(title)
    return matches[0] if len(matches) == 1 else None


def select_best_dataset(
    rbp: str,
    assay_title: str = "eCLIP",
    organism: Optional[str] = "Homo sapiens",
    assembly: Optional[str] = "hg38",
    biosample_hint: Optional[str] = None,
    include_non_released: bool = False,
    encode_base: str = DEFAULT_ENCODE,
    allow_assembly_fallback: bool = True,
) -> Dict[str, Any]:
    experiments = _search_experiments(
        rbp=rbp,
        assay_title=assay_title,
        include_non_released=include_non_released,
        encode_base=encode_base,
    )
    resolution = {"requested": assay_title, "resolved": assay_title, "method": "exact_provider_query"}
    if not experiments:
        available_experiments = _search_experiments(rbp, None, include_non_released, encode_base)
        titles = sorted({exp["assay_title"] for exp in available_experiments if exp.get("assay_title")})
        resolved = resolve_assay_title(assay_title, titles)
        resolution.update(resolved=resolved, available_titles=titles, method="provider_vocabulary")
        if resolved is None:
            return {"selected": None, "candidates": [], "error": "encode_assay_unresolved",
                    "assay_resolution": resolution, "notes": ["Requested assay could not be uniquely bound to the provider's available assay titles."]}
        assay_title = resolved
        experiments = [exp for exp in available_experiments if exp.get("assay_title") == resolved]


    candidates = _collect_candidates(
        experiments,
        rbp=rbp,
        organism=organism,
        assembly=assembly,
        biosample_hint=biosample_hint,
        include_non_released=include_non_released,
        encode_base=encode_base,
        strict_assembly=True,
    )
    notes: List[str] = []
    if not candidates and assembly and allow_assembly_fallback:
        candidates = _collect_candidates(
            experiments,
            rbp=rbp,
            organism=organism,
            assembly=assembly,
            biosample_hint=biosample_hint,
            include_non_released=include_non_released,
            encode_base=encode_base,
            strict_assembly=False,
        )
        if candidates:
            notes.append(f"No peak files matched assembly {assembly!r}; fallback to any assembly.")

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[0] if candidates else None
    return {"selected": selected, "candidates": candidates, "notes": notes, "assay_resolution": resolution}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query ENCODE for an RBP and choose the best eCLIP peak dataset."
    )
    parser.add_argument("rbp", help="RBP target label (e.g. RBFOX2)")
    parser.add_argument("--assay", default="eCLIP", help="ENCODE assay_title (default: eCLIP)")
    parser.add_argument(
        "--organism",
        default="Homo sapiens",
        help="Organism scientific name to prefer (default: Homo sapiens)",
    )
    parser.add_argument("--assembly", default="hg38", help="Assembly to prefer (default: hg38)")
    parser.add_argument(
        "--biosample-hint",
        help="Prefer experiments whose biosample term name contains this string",
    )
    parser.add_argument(
        "--include-non-released",
        action="store_true",
        help="Include non-released experiments/files",
    )
    parser.add_argument(
        "--no-assembly-fallback",
        action="store_true",
        help="Do not fall back to other assemblies when no match is found",
    )
    parser.add_argument(
        "--encode-base",
        default=DEFAULT_ENCODE,
        help="Override ENCODE base URL (default: https://www.encodeproject.org)",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument(
        "--top-only",
        action="store_true",
        help="Only emit the selected dataset (omit candidate list)",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path (defaults to stdout)",
    )
    args = parser.parse_args()

    try:
        result = select_best_dataset(
            rbp=args.rbp,
            assay_title=args.assay,
            organism=args.organism,
            assembly=args.assembly,
            biosample_hint=args.biosample_hint,
            include_non_released=args.include_non_released,
            encode_base=args.encode_base,
            allow_assembly_fallback=not args.no_assembly_fallback,
        )
    except requests.HTTPError as err:
        print(f"HTTP error: {err}", file=sys.stderr)
        if err.response is not None:
            try:
                payload = err.response.json()
            except ValueError:
                payload = err.response.text
            print(payload, file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as err:
        print(f"Request failed: {err}", file=sys.stderr)
        sys.exit(1)

    if args.top_only:
        payload: Any = result.get("selected")
    else:
        payload = result

    if args.pretty:
        content = json.dumps(payload, indent=2, sort_keys=True)
    else:
        content = json.dumps(payload)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(content + "\n")
    else:
        print(content)


if __name__ == "__main__":
    main()
