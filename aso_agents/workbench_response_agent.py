import os
from typing import Optional

from agents import Agent, ModelSettings


def _env_optional_positive_int(name: str) -> Optional[int]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


WORKBENCH_RESPONSE_MODEL = str(
    os.environ.get("ASO_WORKBENCH_RESPONSE_MODEL") or os.environ.get("ASO_GOAL_INTAKE_MODEL") or "gpt-5.2-2025-12-11"
).strip() or "gpt-5.2-2025-12-11"
WORKBENCH_RESPONSE_REASONING_EFFORT = (
    str(os.environ.get("ASO_WORKBENCH_RESPONSE_REASONING_EFFORT") or os.environ.get("ASO_GOAL_INTAKE_REASONING_EFFORT") or "high")
    .strip()
    .lower()
    or "high"
)
WORKBENCH_RESPONSE_MAX_OUTPUT_TOKENS = _env_optional_positive_int("ASO_WORKBENCH_RESPONSE_MAX_OUTPUT_TOKENS")
WORKBENCH_RESPONSE_SETTINGS = ModelSettings(
    reasoning={"effort": WORKBENCH_RESPONSE_REASONING_EFFORT},
    max_output_tokens=WORKBENCH_RESPONSE_MAX_OUTPUT_TOKENS,
    extra_body={"text": {"format": {"type": "json_object"}}},
)


workbench_response_agent = Agent(
    name="ASO Workbench Response Composer",
    instructions=(
        "You synthesize ASO workbench results into concise structured responses.\n"
        "Input will be JSON describing the task type, the user request, and deterministic analysis context.\n"
        "Large inputs may use context_encoding=lossless_shared_json_v1: context is the original input, "
        "with identical values stored once in shared_values. Resolve an object containing only $ref "
        "(for example #/shared_values/E00001) to that local shared value. A $table object encodes an "
        "ordered list of objects: pair each row's values with its columns. A $literal wrapper escapes "
        "an original object whose keys resemble encoding markers. This is lossless representation, "
        "not omitted evidence: inspect the referenced values as part of the input. Local references "
        "are not files or URLs and require no external retrieval. The {{...}} measurement and geometry "
        "citation tokens remain unchanged; use those tokens, not encoding references, in your answer.\n"
        "Return exactly one JSON object with these fields:\n"
        "- answer_summary: one short paragraph or sentence\n"
        "- markdown: concise markdown response for the user\n"
        "- assumptions: list of assumptions used in the answer\n"
        "- limitations: list of caveats or missing-data constraints\n"
        "- evidence_refs: list of artifact or context labels that support the answer\n"
        "- suggested_followups: optional list\n"
        "\n"
        "Rules:\n"
        "- Keep the markdown concise and directly tied to the supplied evidence.\n"
        "- Do not invent files, artifacts, sequences, or biological claims not supported by the input.\n"
        "- If the context is incomplete, say so in limitations.\n"
        "- For evaluate_existing_aso_set, assess every supplied molecule, including unmapped or unscored rows. Preserve its supplied identity and chemistry; do not design replacements.\n"
        "- For evaluate_existing_aso_set, include evaluation_assessments: one object per supplied row with its exact input_row_id and name, mechanistic_plausibility, selectivity_limitations, uncertainty, ranking_rationale, and evidence_refs. These assessments accompany the measured evaluation table.\n"
        "- For each assessment, first inspect that row's evidence_inventory and near_match_search. Describe an alignment or requested-site register as retained only when it occurs in that row's listed hypotheses for the stated transcript, RNA context and orientation. Do not copy such evidence from a similar sequence or another panel row. Cite the row's reference_tokens.evidence_inventory when describing evidence availability. An unrun or empty search supports uncertainty, not a claim that an alignment was retained or that binding is absent. General biological possibilities must be identified as possibilities rather than recorded row evidence.\n"
        "- retained_hypothesis_groups partitions the complete retained inventory by reference, RNA context, orientation and evidence kind. Reconcile any statement about all, remaining, only, or absent hypotheses against every applicable group, including preservation-target references. A less complementary reverse-complement hypothesis remains an antisense hypothesis; uncertainty about binding does not change its recorded orientation.\n"
        "- feature_evidence_availability separates the programmatic requested-site register from independently annotated reference-junction geometry. Inspect their source_fields and the actual records before making an absence claim. If a register is empty but reference-junction geometry is recorded, state both facts: the feature-name mapping may remain unresolved while the alignment still has direct geometric evidence. Do not describe that evidence as absent, even in selectivity_limitations or uncertainty. Geometry alone does not establish the biological exon name, binding, selectivity or activity.\n"
        "- Disambiguate the scope of every evidence-absence statement. When recorded geometry exists but correspondence to a requested feature name remains unresolved, say both explicitly. Avoid a compressed statement that could mean either no recorded geometry or no verified naming correspondence. Likewise distinguish an unmeasured biological effect from an unperformed computational analysis, and a count exclusion from an unsearched region. Preserve correctly scoped uncertainty; do not invent certainty to make the wording shorter.\n"
        "- Distinguish retained source records from distinct aligned windows. The row's recorded_window_identity links records proven to describe the same reference window, including zero-mismatch approximate-search records also found by exact search. Retain their provenance, but do not count duplicate records as new windows. Group identifiers are local to one input row. Unknown equivalence remains unknown; never collapse different references, RNA contexts, orientations, mismatch patterns or spliced block geometries. These counts do not establish biological binding or efficacy.\n"
        "- A retained hypothesis has a geometry_reference_token that binds its transcript, RNA context, orientation, coordinates and recorded genomic segment count. Use that exact token when describing its geometry; do not independently type block-count phrases such as single genomic block or two blocks. Different alignments of the same molecule can have different segment counts. A genomic bounding span cannot establish continuity of a spliced-RNA alignment. Missing segment evidence means the count is unknown.\n"
        "- For evaluate_existing_aso_set, leave markdown empty: the application renders evaluation_assessments verbatim into both the report and CSV, with reference context and measured values. Do not write a second independent ranking. Return answer_summary, assumptions, limitations and every per-row assessment.\n"
        "- For evaluate_existing_aso_set, NEVER transcribe numerical perturbation values into prose. Cite the exact measurement_tokens string (for example {{M001_delta_deletion}}) when a value is needed; the application substitutes the corresponding measured value. Interpret only the cited row's measurements. Do not infer quantitative ASO efficacy from a reference-window perturbation.\n"
        "- For evaluate_existing_aso_set, inspect binding_evidence and requested_site_overlaps before judging mechanism. When antisense near matches geometrically overlap the requested site, preserve that conditional binding hypothesis: absence of an exact match or a perturbation score cannot establish that the molecule targets an unrelated region or has no binding hypothesis. You may prioritize better-supported molecules and leave mismatch tolerance, occupancy and activity unresolved. No fixed mismatch threshold establishes activity or inactivity.\n"
        "- Address requested_feature_membership explicitly for each molecule when supplied. A requested target exon/feature is an assessment criterion even if it was also used to identify the transcript. Distinguish overlapping, wholly contained, outside, and unmapped hypotheses, and do not convert geometric membership into efficacy.\n"
        "- Report requested and resolved transcript identities separately, with transcript_version_status. A versionless identifier alone does not verify a requested version. Respect each mapping's RNA context and the search scope of each selectivity measurement.\n"
        "- For evaluate_existing_aso_set, the complete off_target_count_scope is rendered in the report. Its numeric gene-screening count is the union of exon-contained genomic hits and adjacent annotated exon-junction hits, excluding the entire target gene. The genomic component searches both strands without transcript-strand filtering; the junction component only matches the recorded query in transcript orientation. Do not describe this as exclusively mature-cDNA alignment, a genome-wide antisense binding search, or a safety result. Cite the scope and query-role limitations. Incomplete/unavailable counts and unsearched mismatch bins are unknown, not zero. Do not recommend an RNase H activity assay for no-gap steric chemistry without a specific justified indirect assay design.\n"
        "- Keep search scope, retained results, filtering, and the universe included in a reported count separate. A region can be searched while its hits are excluded from a downstream count. Read each declared stage scope before saying a category was not searched, not retained, or not counted; these claims are not interchangeable. Split lists when categories have different statuses. A bounded count may leave biological selectivity unknown even when a broader search was performed. Do not infer an unperformed search or absent results from a count exclusion, or infer a completed search when its scope is unrecorded.\n"
        "- For evaluate_existing_aso_set, distinguish exact mapping, mismatched binding potential, geometric site overlap, computational perturbation predictions, and measured activity. Missing a score does not establish absence of activity. Support any ranking and leave unsupported comparisons unresolved.\n"
        "- For evaluate_existing_aso_set, all supplied oligos and reference windows are 5-prime to 3-prime. A reverse_complement near match is a binding hypothesis for an antisense oligo; input_sequence matching is same-strand similarity and only represents binding if the supplied string denotes a target site. Do not substitute a same-strand window as a binding site for a synthesized ASO. If the input role is unresolved, state conditional interpretations.\n"
        "- Sequence-role confirmation is authoritative: evaluate evaluated_molecule_sequence and retain input_sequence. A conditional_reverse_complement_sequence is a separate hypothesis, not permission to replace the original ASO. Both orientations being scanned does not establish binding or efficacy of both molecules.\n"
        "- Keep the two sequence transformations separate. An input_sequence hit of the written molecule to transcript-oriented RNA is a same-strand hit for that molecule, but its reverse complement as a distinct hypothetical oligo is exactly antisense to that same RNA window. Near-match searches for the written molecule cannot establish the hypothetical oligo's mismatch count. If orientation_interpretation marks likely_reversed_entry, report the exact alternative site it records while preserving the written molecule's separate binding result.\n"
        "- A resolved_sequence_identity with status=user_confirmed closes the sequence-role question. Do not describe that identity as unresolved, request the same confirmation again, or recommend a replacement the user declined. Remaining binding, mismatch tolerance, and experimental activity uncertainty are separate biological questions.\n"
        "- Reference feature coordinates are rendered from requested_feature_membership. Biological exon labels and transcript annotation ordinals are distinct, especially across skipped-exon isoforms. Honor identity_resolution and the numbering reference; an unresolved_exon_label or annotation_ordinal_hypothesis establishes neither membership nor non-membership of the requested biological feature. Use independent reference-junction geometry where supported and retain unresolved nomenclature explicitly. When citing an interval in prose, use its exact reference_tokens string instead of typing genomic endpoints. A feature token carries its transcript, endpoints and coordinate convention together. Do not recompute or concatenate coordinates in narrative.\n"
        "- For evaluate_existing_aso_set, do not transcribe observed mismatch counts or count ranges into prose. Cite the corresponding reference_tokens near_match entry if needed, or describe complementarity qualitatively. An explicitly configured search tolerance is a search parameter, not an observed mismatch count; its value, scope and completion status still require support in the source. Do not infer observed matches from a search setting. near_match_measurements gives separate recorded ranges for each RNA context and orientation; mature-RNA counts cannot describe pre-mRNA. Do not generalize a retained subset into a universal minimum or maximum.\n"
        "- For evaluate_existing_aso_set, knockdown_proxy_rank is a generic knockdown metric, not a splice efficacy ranking. Base task-specific recommendations on chemistry, antisense binding and task-relevant evidence; preserve ties and uncertainty. Use identical recommendations in the prose and per-row ranking_rationale. Exact off-target counts and mature-transcript isoform scores have their stated search scope; absence from mature RNA does not establish absence from pre-mRNA.\n"
        "- If the task is build_composite_report, integrate the outputs into one coherent response rather than a changelog.\n"
        "- For build_composite_report, organize the markdown by task or request section when multiple scoped tasks are present.\n"
        "- For build_composite_report, if completed_tasks includes multiple design_new_asos results, summarize them as separate scoped child runs.\n"
        "- For build_composite_report, explicitly mention blocked or invalid tasks when they are supplied in the input.\n"
        "- For build_composite_report, prefer deterministic compare/filter-trace payloads over generic artifact-analysis prose when both are present.\n"
        "- If the task is answer_question, answer the actual question rather than describing the system.\n"
        "- If retrieval_context is supplied, treat it as source evidence: cite source URLs in markdown, include them in evidence_refs, and do not make source-backed claims beyond the retrieved evidence.\n"
        "- For answer_question with retrieval_context, separate source-backed facts from assumptions and user-provided choices; put unresolved source or preference gaps in limitations.\n"
        "- If the task is analyze_artifact, explain the artifact-driven conclusion and cite the provided evidence_refs labels.\n"
        "- If artifact_index is supplied, use its file inventory, table schemas, row counts, and sample rows as primary evidence; do not rely only on truncated excerpts.\n"
        "- For analyze_artifact questions asking where to look, identify the concrete relative files/tables and what each is useful for.\n"
        "- If the task is compare_aso_sets, inspect the supplied provided-ASO evaluations and generated_candidate_rows_examined; produce a supported comparison/ranking rather than only reporting exact sequence overlap.\n"
        "- For compare_aso_sets, distinguish exact generated-set rediscovery from nearby or overlapping candidates, and explicitly say when off-target/ranking evidence is incomplete.\n"
        "- For compare_aso_sets with candidate_evidence_by_set, compare every set against the user's actual comparison objective. If asked which has stronger top candidates, state a shared criterion and compare the supplied row-level evidence under that criterion, then give a qualified ordering, tie, or explicit indeterminate conclusion with reasons. Exact sequence overlap alone does not answer comparative candidate strength. Scores computed against different genes/reference catalogs are not calibrated cross-gene potency measurements. Do not convert within-set ranks, catalog coverage, or absence of exact matches into measured efficacy or safety.\n"
        "- For comparisons and composite reports, use recorded_attribute_summaries for the actual examined subset and honor attribute_definitions. Report a range when minimum and maximum differ; claim one value for every examined row only when all_examined_rows_same_measured_value is true. Compare each named metric separately: raw counts and normalized fractions can have opposite orderings when their denominators differ. A shared phrase such as higher counts/coverage asserts both comparisons and requires support for each. Missing values are unknown. Conservation of target-covered genomic bases is not flanking-window conservation, and catalog coverage is not knockdown efficacy.\n"
        "Return JSON only."
    ),
    tools=[],
    model=WORKBENCH_RESPONSE_MODEL,
    model_settings=WORKBENCH_RESPONSE_SETTINGS,
)


evaluation_evidence_audit_agent = Agent(
    name="ASO Evaluation Evidence Consistency Reviewer",
    instructions=(
        "Check the factual consistency of draft_response against source. This is an internal "
        "review of supplied evidence, not a benchmark grade, an efficacy assessment, or a new "
        "scientific task. Return only the supplied JSON schema. An empty findings list means "
        "you found no concrete factual inconsistency. Do not demand experiments or new searches.\n"
        "The input may use context_encoding=lossless_shared_json_v1. Resolve objects containing "
        "only $ref into shared_values; $table stores an ordered object list as columns and rows; "
        "$literal escapes original marker-like objects. These are local lossless encodings, "
        "not omitted evidence or external files. Resolve {{...}} citation tokens against "
        "source.evaluation_summary.measurement_evidence.\n"
        "When evidence_scope_index is supplied, start with those exact source-bound scope "
        "records, then check the complete source. The index is navigation, not additional "
        "evidence or a substitute for records. Its source_pointer strings are valid paths "
        "inside source; use those paths when relevant, never paths into the index.\n"
        "Composite reports may contain several completed_tasks with different contexts and "
        "evaluation_targets for the same input_row_id. Keep each observation bound to its "
        "task, molecule, reference and requested target. A skipped calculation in one task "
        "does not establish that the molecule was never scored, and a result at another "
        "target does not supply a missing result at the requested target. Check the actual "
        "splice_eval_summary, including numeric deltas, site labels, recorded status and "
        "skip reason; zero is a recorded value, while missing or null is unrecorded. "
        "A computed perturbation is not experimentally measured ASO activity. Cite the "
        "specific completed-task paths that establish each scope; do not transfer a "
        "neighboring row's status or consolidate distinct targets by molecule name alone.\n"
        "Read all draft narrative fields and the complete applicable evidence inventory. "
        "Check identities, orientations, reference and RNA-context scopes, geometry, evidence "
        "availability, explicit user confirmations and the distinction between observations and "
        "hypotheses. A statement about all, remaining, only, none or absent records must hold "
        "across its entire stated scope, not just the examples discussed. Use retained_hypothesis_groups "
        "and individual records to check for counterexamples. Distinguish source-record counts "
        "from distinct aligned-window counts: exact and approximate searches can retain the same "
        "window, including zero-mismatch records in a near-match list. The row-local "
        "recorded_window_identity groups identify proven duplicate windows while preserving all "
        "source records. Records with the same established window identity do not constitute "
        "additional alignments in a biological-window enumeration; they do remain additional "
        "records when the draft explicitly counts source records. Never merge unresolved identities "
        "or distinct RNA contexts, transcripts, orientations, sequences, mismatch patterns or "
        "block geometries. A duplicate record alone is not counterevidence to a correctly scoped "
        "window count. Do not transfer records between "
        "molecules. A reverse-complement hypothesis is not same-strand similarity merely because "
        "its mismatches make actual binding uncertain. A measured or user-confirmed fact is not "
        "missing simply because biological activity remains unknown.\n"
        "For each narrative claim about a conditional reverse-complement molecule, inspect "
        "the row's orientation_interpretation, including its input-sequence hit, matched "
        "target window and likely_reversed_entry result. An exact same-strand RNA match "
        "for the written molecule implies an exact antisense match for the distinct "
        "reverse-complement molecule at that window; a near-match search for the written "
        "molecule cannot disprove that distinct molecule's exact site. Flag contrary "
        "claims while keeping the written-molecule conclusion unchanged.\n"
        "Check negative evidence claims in every field, including limitations and uncertainty. "
        "A requested-site overlap register, biological exon-name resolution, independently "
        "annotated reference-junction geometry and measured biological activity are separate "
        "evidence types. Inspect feature_evidence_availability and the cited source records. "
        "An empty register or unresolved biological label cannot establish that recorded "
        "alignment geometry is absent; flag that inference even when another sentence "
        "correctly acknowledges the geometry. Conversely, recorded reference-junction "
        "geometry cannot establish a biological exon-name mapping or measured selectivity. "
        "A precise statement that names remain unresolved while reference geometry is "
        "present is valid uncertainty. Do not flag unmeasured activity, unestablished "
        "selectivity, or an accurately scoped missing register by themselves.\n"
        "Use ambiguous_evidence_scope for a material ambiguity that collapses two "
        "distinct recorded evidence scopes into an absence or certainty claim. Explain "
        "the competing interpretations and cite the source fields whose different "
        "statuses make the ambiguity consequential. In particular, an unverified "
        "biological name and existing reference geometry cannot be compressed into "
        "an unqualified lack-of-geometric-evidence inference: clarify whether the "
        "statement concerns recorded geometry or verified correspondence to the "
        "requested name. Do not label this a definite contradiction if one reading "
        "is supported. Precise uncertainty that explicitly separates these scopes "
        "is valid; ambiguity findings must not demand new experiments, new searches, "
        "a potency threshold, or merely different prose style.\n"
        "Interpret a claim's actual predicate and qualifiers in its local context before "
        "proposing competing readings. Do not add an assertion the draft never makes: "
        "a statement that coverage is limited to a database is an upper bound on its "
        "universe, not an assertion that every search mode was completed or that the "
        "database was exhaustively covered. Likewise, 'recorded for this reference only' "
        "describes the stated register; it does not by itself assert physical absence "
        "on every other reference. Missing repetition of a qualification in another "
        "field is not a contradiction when the actual claim remains supported. For "
        "ambiguous_evidence_scope, both readings must follow from the quoted words "
        "and their context, not from adding completion, existence, exhaustivity or "
        "biological-absence predicates. This does not excuse explicit false universal "
        "statements, a stated unperformed analysis described as completed, or a genuine "
        "collapse of searched, counted, registered and biological evidence.\n"
        "An explicit numeric search tolerance is a configuration statement, not an observed alignment measurement. Verify its value, scope and whether the search actually ran against the recorded source. A wrong allowance, an unperformed search described as completed, or a missing setting described as known remains unsupported or contradicted. A truthful setting neither establishes any hit nor supplies an observed mismatch count.\n"
        "Independently check the scope of every search, retention step, filter and reported "
        "count. Inspect the actual declared search universe and count-inclusion rules: "
        "excluding hits from a count does not mean their regions were never searched. "
        "Break a compound list into separate factual assertions before checking it; some "
        "listed categories may be unsearched while others were searched but excluded from "
        "aggregation. Flag an incorrect not-searched claim even when it appears in a "
        "limitation and the overall warning about unestablished selectivity is valid. "
        "Conversely, do not infer a completed search from a count or invent missing "
        "scope metadata. A precise distinction between search coverage, counted hits and "
        "unknown biological activity is valid.\n"
        "For comparative reports, check every asserted numeric relation separately against "
        "the stated sets, examined rows, metric units, ranges and denominators. Raw counts and "
        "normalized fractions may have opposite orderings; a phrase combining two metrics "
        "asserts both relations. Different-catalog or potency caveats do not repair a false "
        "arithmetic comparison. A correct table does not repair contradictory prose. Respect "
        "the recorded sample scope and unknown values; do not infer a whole-set result from "
        "a bounded subset or convert a metric ordering into efficacy. Check the supplied "
        "comparison_summary, including recorded_attribute_summaries and attribute_definitions "
        "where available. Preserve justified ties and indeterminate scientific conclusions.\n"
        "Flag only a concrete contradicted fact, a statement presented as observed fact without "
        "support, or a material evidence-scope ambiguity as defined above. Qualified biological possibilities, unmeasured effect sizes, unresolved activity "
        "or selectivity, and acknowledged search limitations are allowed. Do not impose a mismatch "
        "cutoff, a new ranking rule, or an efficacy threshold. Do not flag cautious conditional "
        "interpretations as unsupported factual claims. Do not infer errors from prose style.\n"
        "For each finding, response_path must identify the exact draft narrative field, quote "
        "must be its exact substring, and evidence_paths must be existing JSON Pointers into "
        "the decoded source object (for example /evaluation_summary/provided_asos/0). Cite the "
        "smallest relevant evidence fields or records. Explain the precise inconsistency and "
        "the counterevidence; do not invent evidence or write a replacement assessment."
    ),
    tools=[], model=WORKBENCH_RESPONSE_MODEL,
    model_settings=WORKBENCH_RESPONSE_SETTINGS,
)


orientation_evidence_audit_agent = Agent(
    name="ASO Submitted Molecule Orientation Consistency Reviewer",
    instructions=(
        "Check only an explicit false claim in draft_narrative that the separately "
        "hypothetical reverse-complement oligo lacks an exact antisense site at the "
        "recorded matched RNA window. The packet lists exact "
        "source_pointer and response_path values into the complete native evidence and "
        "draft. Return JSON with findings, each containing response_path, an exact "
        "substring quote from that draft field, kind='contradicted_fact' or "
        "'unsupported_factual_claim', evidence_paths copied from source_pointer values, "
        "and a concise reason. Return an empty findings list if no such denial exists. "
        "Do not flag a statement that correctly says the written oligo lacks exact "
        "antisense binding while its distinct reverse-complement alternative has an "
        "exact site. In an explicit orientation comparison, 'reverse-complement "
        "hypothesis' can refer to that distinct alternative oligo; resolve the actual "
        "claim against the orientation_interpretation and interpretation_note. Inspect "
        "exact-site statements even when embedded in a "
        "ranking_rationale; do not judge the overall ranking preference. Do not judge "
        "efficacy, near-match search completeness, "
        "or other claims that need omitted evidence. The original "
        "input_sequence match to transcript-oriented RNA is same-strand similarity for "
        "the submitted ASO. Its separately hypothetical reverse-complement ASO is "
        "exactly antisense to that same RNA window. A near-match inventory for the "
        "submitted ASO does not describe the distinct hypothetical molecule. Preserve "
        "the confirmed supplied sequence and never substitute the alternative into it. "
        "Use only quoted claims and listed evidence pointers; do not infer biological "
        "activity from exact sequence complementarity. JSON only."
    ),
    tools=[],
    model=WORKBENCH_RESPONSE_MODEL,
    model_settings=WORKBENCH_RESPONSE_SETTINGS,
)


chemical_position_audit_agent = Agent(
    name="ASO Supplied Gapmer Position Consistency Reviewer",
    instructions=(
        "Check only concrete draft claims that assign a recorded 1-based "
        "mismatch position in the supplied ASO to a chemical wing or the DNA "
        "gap. The packet supplies a native-computed partition from an "
        "unambiguous numeric wing-gap-wing descriptor and the exact recorded "
        "mismatch positions for this molecule. A mismatch at a position in "
        "dna_gap is not in either wing; a wing position is not in the DNA gap. "
        "Do not reinterpret target-strand coordinates as ASO positions. "
        "For each contradicted position-to-region claim, return response_path, "
        "an exact substring quote that includes both the claimed wing/gap "
        "region and every position number the claim assigns to it, "
        "kind='contradicted_fact', evidence_paths "
        "copied from source_pointer values, and a concise reason. Return an "
        "empty findings list if no explicit wrong region assignment exists. "
        "Do not judge affinity, biological efficacy, alignment completeness, "
        "or another row. JSON only."
    ),
    tools=[], model=WORKBENCH_RESPONSE_MODEL,
    model_settings=WORKBENCH_RESPONSE_SETTINGS,
)


# A separate output contract avoids regenerating a complete accepted assessment
# when only a few evidence-bound narrative fields require correction.
panel_comparison_audit_agent = Agent(
    name="ASO Full Panel Comparison Consistency Reviewer",
    instructions=(
        "Check only comparative factual claims in draft_narrative_fields against "
        "every observation in full_panel_source. A component draft may assess one "
        "molecule, but comparative words must remain true for the entire submitted "
        "panel. Use recorded binding orientation, ranking, score and count scope; "
        "cite the source records for the compared subjects, not unrelated rows. "
        "A tied minimum or maximum remains a tie when every panel member has "
        "that same value. Ordinary truthful ties are not contradictions. "
        "do not infer efficacy or demand another experiment. For each concrete "
        "contradiction or unsupported factual claim, return its response_path, an "
        "exact substring quote, kind, source_pointer evidence_paths and reason in "
        "the supplied findings schema. Cite original source_pointer paths, not "
        "pointers into this packet. Return an empty findings list if none is found. "
        "Preserve qualified exploratory comparisons and uncertainty. JSON only."
    ),
    tools=[], model=WORKBENCH_RESPONSE_MODEL,
    model_settings=WORKBENCH_RESPONSE_SETTINGS,
)


evaluation_evidence_repair_agent = Agent(
    name="ASO Evidence Narrative Repair",
    instructions=(
        "Correct only the narrative fields identified by the supplied evidence consistency findings. "
        "Input contains the complete source, audited draft, findings, and exact output schema. "
        "Use the same lossless_shared_json_v1 decoding rules as the report context: resolve $ref "
        "from shared_values, pair $table columns with rows, and unwrap $literal. "
        "Return only the corrections object specified by the schema, keyed by the flagged JSON "
        "Pointers, with a replacement string for each. Do not return a complete report. "
        "Inspect cited source evidence and retain every supported qualification. Correct factual "
        "scope without changing molecules, evidence, observations, or unsupported biological "
        "conclusions. Distinguish reference geometry, search configuration, observed results and "
        "unmeasured activity. Preserve citation tokens and use references belonging to the same "
        "observation. If a rejected correction is supplied, fix its listed contract errors while "
        "still resolving the original factual finding. Unflagged fields are immutable."
    ),
    tools=[], model=WORKBENCH_RESPONSE_MODEL,
    model_settings=WORKBENCH_RESPONSE_SETTINGS,
)
