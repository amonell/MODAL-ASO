"""A late, selection-only size preference; no scientific restart."""
import re

from .candidate_assessment import shortlist_size_specified


def size_question(goal, has_candidates):
    if not has_candidates or shortlist_size_specified(goal):
        return None
    return {"kind":"shortlist_size", "restart_stage":"report",
        "reason":"Candidate scoring is complete; the recommendation-panel size is unspecified.",
        "questions":["How many distinct ASOs would you like in the recommended shortlist? Enter a positive whole number for the complete panel, or 'all eligible'. The full candidate pool will remain available."],
        "context":"Selection will use the existing computed evidence and preserve all region, chemistry and minimum-coverage requirements. An incompatible total will be reported as infeasible rather than silently relaxing them.",
        "why_this_matters":"The requested size controls the recommended panel, not the number of candidates screened.",
        "suggested_answers":["all eligible"] if any(goal.get(key) for key in ("min_asos_per_site", "candidate_minimum")) else ["6", "12", "all eligible"]}


def parse_size_answer(answers):
    texts = [str(item.get("answer") or "").strip() for item in answers if isinstance(item,dict)]
    if len(texts) != 1:
        raise ValueError("Please provide one overall shortlist size, or 'all eligible'.")
    text = texts[0].lower().rstrip(".! ")
    if text in {"all", "all eligible", "all eligible asos", "all eligible candidates", "all eligible molecules"}:
        return {"shortlist_size_preference":"all_eligible"}
    match = re.fullmatch(r"(?:please\s+)?(?:(?:i(?:'d| would) like|give me|recommend|select)\s+)?([1-9][0-9]*)(?:\s+(?:asos|candidates|molecules))?(?:\s+(?:total|overall|please))?", text)
    if not match:
        raise ValueError("Please enter one positive whole number for the complete panel, or 'all eligible'; per-site counts require an explicit request update.")
    count = int(match[1])
    return {"candidate_limit":count, "candidate_minimum":count}
