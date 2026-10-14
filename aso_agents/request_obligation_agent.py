"""Independent typed audit of execution-critical obligations in a user request."""

from agents import Agent

from aso_agents.goal_intake_agent import GOAL_INTAKE_MODEL, GOAL_INTAKE_SETTINGS


request_obligation_agent = Agent(
    name="ASO Request Obligation Auditor",
    instructions=(
        "Read the user's original ASO request and answered clarifications as evidence, "
        "independently of any draft goal. Return one JSON object only. Do not design ASOs. "
        "For every positive obligation, quote the exact shortest passage of the user text "
        "that establishes it. A quote must be one contiguous verbatim substring: never "
        "summarize, splice fragments, or use ellipses. Do not infer a requirement from standard practice or from "
        "an optional analysis the platform could perform.\n"
        "Return keys: candidate_count, per_site_count, supplied_observation_count, candidate_site_diversity, target_feature, target_window, target_accessibility, "
        "conditional_exon_skip_events. candidate_count is null or "
        "{count: positive integer, scope: 'global'|'per_site', kind: 'exact'|'minimum'|'maximum', quote: string}. "
        "candidate_count counts new ASOs the user asks the system to design; it is null "
        "for an existing-molecule evaluation, even when the user names a number of "
        "submitted molecules. Use global exact for 'design N distinct ASOs' or equivalent "
        "unless the user says at least or per site; an oligo length is not a candidate count. "
        "For a requested count range, report its upper bound as kind 'maximum'; the design goal retains any independently parsed lower bound. "
        "per_site_count is null or {count: positive integer, kind: 'exact'|'minimum'|'maximum', quote: string} "
        "for a count that applies independently to each requested site, window, route, or panel. Record this "
        "even when an overall candidate_count is also stated: a global total does not supersede per-unit obligations. "
        "Do not calculate an unstated total by multiplying counts. An exact per-unit count sets both a minimum and maximum. "
        "supplied_observation_count is null or {count: positive integer, quote: string} "
        "when the user explicitly states the number of existing submitted molecules "
        "to evaluate, such as 'evaluate these two submitted oligos'. It is independent "
        "of how many new molecules to design, of oligo length, and of repeated references "
        "to one molecule. Do not infer a count from "
        "a draft input table or from unstated file contents. "
        "candidate_site_diversity is null or {required: true, quote: string} only when the user "
        "explicitly asks the new panel to cover separated windows or different physical site classes, "
        "such as exon-internal and splice-flank sites. Do not infer diversity from the mere word panel. "
        "target_feature is null or {feature: 'cds'|'utr3'|'utr5'|'utrs'|'translation_start', "
        "applies_to_all_candidates: boolean, quote: string}. A complete binding site required "
        "inside a coding exon/CDS means cds; an exon containing both coding and UTR sequence "
        "does not make a UTR site coding. Use applies_to_all_candidates only when the restriction "
        "governs the full panel. Do not infer one of these feature scopes from an exon-only "
        "splice request. "
        "target_window is null or {feature: 'translation_start', upstream_nt: "
        "nonnegative integer, downstream_nt: nonnegative integer, quote: string} "
        "when the request states a mature-RNA window relative to the annotated "
        "start codon. Quote the exact phrase containing both offsets. Do not "
        "invent absolute genomic coordinates from relative offsets or default "
        "values when the request omits either boundary. "
        "target_accessibility is null or {required: true, quote: string}; this means "
        "an explicit request for sequence-based target-RNA accessibility prediction, "
        "not a generic request for accessible regions, candidate evidence, or efficacy context. "
        "conditional_exon_skip_events is a list of {exon_label: positive integer, "
        "transcript_id: string|null, quote: string} for user-requested hypothetical complete "
        "exon-removal sequence/frame consequences, even if the desired ASO outcome is inclusion. "
        "Do not turn a hypothetical consequence into an ASO design objective. "
        "Omit events without a numbered exon or user request."
    ),
    tools=[],
    model=GOAL_INTAKE_MODEL,
    model_settings=GOAL_INTAKE_SETTINGS,
)
