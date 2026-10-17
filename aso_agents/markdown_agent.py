from typing import Dict, Any
from agents import Agent

markdown_agent = Agent(
    name="ASO Markdown Reporter",
    instructions=(
        "You produce a concise, decision-focused Markdown summary of the ASO pipeline run.\n"
        "Input is a JSON object with keys including: structured_goal, transcript_recommendation, bed, chemistry, splicing, splicing_eval, tiling, tiles_json_path, tiles_with_deltas_path, filtered_tiles_path, ranked_tiles_path, ranking, general_rationale, constraint/filter summaries, strategy_routing (if present), and branch_outcomes.\n"
        "Required sections (keep tight):\n"
        "1) Goal & intent: gene, intent (knockdown vs splice_modulation), tissues/context, any PTC/NMD or exon targets.\n"
        "If pipeline_alerts is present and non-empty, add a short 'Warnings' section near the top listing the blockers in plain language.\n"
        "If branch_outcomes is present, state branch-level no-candidate outcomes bluntly. For status='no_candidates_after_constraints', say the requested constraints exhausted that branch's search space. For status='no_candidates_found', say the branch ran but produced no qualifying candidates.\n"
        "2) Transcript decision: chosen ID and one-line rationale (agent decision).\n"
        "3) Splicing plan & strategy: key junctions/indices with directions; if splice_strategy_analysis present, distill recommended junctions and backups; otherwise, if structured_goal.splice_site_plan_preview is present, summarize that preview instead of saying no splice junctions were selected. If branch_outcomes.steric.status='no_candidates_after_constraints', say splice-window planning existed but the requested steric constraints exhausted the search space before routing. If strategy_routing present, explicitly state that route ranking was LLM-decided, name the ranking model when provided, and include a ranked sublist of routes in order with goal label, rationale, count, threshold used, and csv/output paths when present. Note what the route CSV columns represent (primary_label/threshold/targets) without embellishment. If a route goal has status='no_candidates_found' or count=0, say explicitly that no candidates were found for that goal and do not phrase it as a successful ranked output.\n"
        "4) SpliceTransformer eval: note sites scored, non-zero vs all-zero deltas, and which JSON path was scored.\n"
        "5) BED/steric regions: counts, labels, and path.\n"
        "6) Tiling & filters: config path, snakemake cmd in a short code fence, counts per filter bucket, constraints enforced (GC/homopolymer/max binding if present).\n"
        "7) Chemistry: recommended chemistry text/constraints.\n"
        "8) Ranking/selection: top-ranked items with ids/positions, filtered/kept counts, and any rationale provided.\n"
        "Use small fenced snippets only where helpful (e.g., snakemake_cmd).\n"
        "Do NOT add suggestions or next steps. Output only Markdown."
    ),
    tools=[],
    model="gpt-5-mini-2025-08-07",
)
