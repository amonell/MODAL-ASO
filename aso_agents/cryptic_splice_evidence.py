"""Scoped unannotated splice-score gains from already computed perturbations.

No score threshold here diagnoses a cryptic splice product. Report the largest
gain and search scope rather than inventing a biological pass/fail cutoff.
"""
import hashlib
import json

import numpy as np


def annotation_sites(exons, strand, supplied_sites=()):
    points = {"acceptor":set(),"donor":set()}
    minus = strand == "-"
    ordered = sorted(exons or [],key=lambda exon:exon["start"],reverse=minus)
    for index,exon in enumerate(ordered):
        if index > 0:
            points["acceptor"].add(exon["end"] if minus else exon["start"])
        if index + 1 < len(ordered):
            points["donor"].add(exon["start"] if minus else exon["end"])
    for site in supplied_sites:
        for kind in points:
            value = (site.get(kind) or {}).get("position") if isinstance(site.get(kind),dict) else site.get(kind)
            if type(value) is int:
                points[kind].add(value)
    return {key:sorted(values) for key,values in points.items()}


def unannotated_gain(reference, perturbed, *, chrom, left, right, strand,
                     annotated, annotation_radius_nt=0, excluded_intervals=()):
    context = {"chrom":chrom,"start_1based":left,"end_1based":right,"strand":strand,
        "annotation_scope":"resolved transcript exons and supplied evaluation sites",
        "annotation_radius_nt":annotation_radius_nt,
        "annotation_sha256":hashlib.sha256(json.dumps(annotated,sort_keys=True).encode()).hexdigest(),
        "excluded_intervals_1based_inclusive":list(excluded_intervals)}
    result = {"status":"unavailable","scope":context,"sites":{},
        "interpretation":"Largest model-score gains outside the declared annotation and perturbed ASO span. These are review flags, not confirmed cryptic sites or absence-of-cryptic-splicing evidence. Predictions use the recorded ASO perturbation approximation."}
    if strand not in {"+","-"} or type(left) is not int or type(right) is not int or left < 1 or right < left:
        result["reason"] = "invalid_genomic_frame"
        return result
    if type(annotation_radius_nt) is not int or annotation_radius_nt < 0:
        result["reason"] = "invalid_annotation_radius"
        return result
    ref,alt = np.asarray(reference),np.asarray(perturbed)
    length = right-left+1
    if ref.ndim != 2 or alt.shape != ref.shape or ref.shape[0] != length or ref.shape[1] < 3:
        result["reason"] = "prediction_arrays_do_not_match_genomic_frame"
        return result
    positions = np.arange(left,right+1) if strand == "+" else np.arange(right,left-1,-1)
    base = np.ones(length,dtype=bool)
    for start,end in excluded_intervals:
        base &= (positions < start) | (positions > end)
    incomplete = False
    for column,kind in ((1,"acceptor"),(2,"donor")):
        eligible = base.copy()
        for position in annotated.get(kind,[]):
            index = position-left if strand == "+" else right-position
            lower,upper = max(0,index-annotation_radius_nt),min(length,index+annotation_radius_nt+1)
            if lower < upper:
                eligible[lower:upper] = False
        finite = np.isfinite(ref[:,column]) & np.isfinite(alt[:,column])
        incomplete |= bool(np.any(eligible & ~finite))
        eligible &= finite
        count = int(np.count_nonzero(eligible))
        summary = {"scanned_position_count":count,"positive_gain_position_count":0,"largest_gain":None}
        if count:
            indices = np.flatnonzero(eligible)
            deltas = alt[indices,column].astype(float)-ref[indices,column].astype(float)
            maximum = float(np.max(deltas))
            summary["positive_gain_position_count"] = int(np.count_nonzero(deltas > 0))
            summary["max_delta"] = maximum
            if maximum > 0:
                ties = indices[deltas == maximum]
                best = int(ties[np.argmin(positions[ties])])
                summary["largest_gain"] = {"position_1based":int(positions[best]),
                    "nearest_annotated_distance_nt":min((abs(int(positions[best])-position) for position in annotated.get(kind,[])),default=None),
                    "reference_score":float(ref[best,column]),"perturbed_score":float(alt[best,column]),
                    "delta":maximum,"tied_position_count":len(ties)}
        result["sites"][kind] = summary
    result["status"] = ("incomplete" if incomplete else "scanned_within_declared_scope"
                        if any(site["scanned_position_count"] for site in result["sites"].values())
                        else "no_scannable_positions")
    return result


def paired_evidence(reference_deletion, deletion, reference_replacement, replacement,
                    *, model, first_perturbation, **context):
    return {"model":model,"first_perturbation":first_perturbation,
        "deletion":unannotated_gain(reference_deletion,deletion,**context),
        "replacement":unannotated_gain(reference_replacement,replacement,**context)}
