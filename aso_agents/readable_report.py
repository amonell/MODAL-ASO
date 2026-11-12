"""Reader-facing HTML over saved reports; never rewrite the scientific source.

Short explanations introduce verified saved counts and terms. Free scientific
prose, table records, identifiers, units and sequences remain available intact.
Unknown sections stay expanded. Known evidence sections open on demand, with
their interpretation and limitations beside the entry point.
This module deliberately has no model, prediction or selection dependencies.
"""
from __future__ import annotations

import html
import re
import ast

from .report_presentation import TABLE_CSS, markdown_html
from .report_values import (
    FIELD_LABELS, VALUE_CSS, VALUE_LABELS, field_label, identity_key,
    parse_collection, readable_prose, value_html,
)


READER_CSS = TABLE_CSS + VALUE_CSS + """
:root{color-scheme:light;--reader-ink:#24332f;--reader-muted:#53645e;--reader-line:#dbe4de;--reader-accent:#23624e}
*{box-sizing:border-box}
body{margin:0;background:#f7f9f7;color:var(--reader-ink);font:16px/1.65 system-ui,-apple-system,'Segoe UI',sans-serif}
.reader-page{max-width:1120px;margin:auto;padding:32px 28px 64px;min-width:0}
.reader-header{padding:0 0 22px;border-bottom:1px solid var(--reader-line);margin-bottom:24px}
.reader-brand{font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;color:var(--reader-accent);font-weight:700;margin:0 0 8px}
.reader-header h1{font-size:2rem;line-height:1.2;margin:0 0 10px;letter-spacing:-.035em;overflow-wrap:anywhere}
.reader-subtitle{color:var(--reader-muted);margin:0;max-width:75ch}
.reader-content{min-width:0;background:white;border:1px solid var(--reader-line);border-radius:12px;padding:24px 28px}
.reader-content h2{font-size:1.25rem;line-height:1.4;margin:0 0 12px;letter-spacing:-.015em}
.reader-content h3{font-size:1.05rem;line-height:1.5;text-transform:none;letter-spacing:normal;color:var(--reader-ink)}
.reader-content p,.reader-content li{max-width:85ch}
.reader-section{margin:0 0 28px;scroll-margin-top:20px}
.reader-section:last-child{margin-bottom:0}
.reader-section.reader-limits{border-left:3px solid #b07b38;padding-left:18px}
.reader-section.reader-downloads{padding:18px;background:#f1f6f2;border-radius:8px}
.reader-content details{border:1px solid var(--reader-line);border-radius:8px;padding:12px 16px;margin:14px 0}
.reader-content details>summary{cursor:pointer;font-weight:600;list-style-position:outside;margin-left:16px}
.reader-content details[open]>summary{margin-bottom:16px}
.reader-content details>div{min-width:0}
.reader-content a{color:var(--reader-accent);text-underline-offset:3px}
.reader-nav{display:flex;gap:8px 20px;flex-wrap:wrap;margin:14px 0 24px}
.reader-nav a{font-size:.9rem;color:var(--reader-accent)}
.reader-tools{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin:0 0 18px}
.reader-tools p{font-size:.88rem;color:var(--reader-muted);margin:0}
.reader-tools button{font:inherit;font-size:.9rem;cursor:pointer;border:1px solid var(--reader-line);border-radius:6px;background:white;padding:6px 12px;color:var(--reader-accent)}
.reader-glossary dl{display:grid;grid-template-columns:minmax(120px,1fr) 3fr;gap:10px 18px;margin:0}
.reader-glossary dt{font-weight:600}.reader-glossary dd{margin:0}
.reader-content .table-scroll{border-color:var(--reader-line);border-radius:6px}
.reader-content th{background:#f1f6f2}
.reader-content td,.reader-content th{border-color:var(--reader-line);font-size:.9rem;padding:10px 12px}
.reader-content code,.reader-content pre{background:#f3f5f3}
.reader-content .sequence{white-space:normal;overflow-wrap:anywhere}
.reader-content td details.reader-rank{border:0;padding:0;margin:0}
.reader-content .reader-rank summary{white-space:nowrap;font-weight:400}
.reader-content .reader-rank code{white-space:normal;overflow-wrap:anywhere}
.reader-table-compact .reader-detail-column{display:none}
.reader-column-toggle{font:inherit;font-size:.88rem;border:1px solid var(--reader-line);background:white;color:var(--reader-accent);border-radius:6px;padding:6px 10px;cursor:pointer}
.reader-orientation{border-bottom:1px solid var(--reader-line);padding:0 0 24px;margin:0 0 28px}
.reader-eyebrow{color:var(--reader-accent);font-size:.76rem;letter-spacing:.09em;text-transform:uppercase;font-weight:700;margin:0 0 6px}
.reader-orientation h2{font-size:1.55rem;margin:0 0 10px}.reader-orientation h3{font-size:1rem;margin:16px 0 6px}
.reader-purpose{font-size:1.05rem;margin:0 0 18px}
.reader-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0;max-width:900px}
.reader-fact{display:flex;flex-direction:column;align-items:flex-start;background:#f2f6f2;border-radius:8px;padding:16px;min-width:0}
.reader-fact-value{font-size:1.9rem;font-weight:650;line-height:1.2;color:#244f40;overflow-wrap:anywhere}
.reader-fact-label{font-weight:600;margin-top:8px}.reader-fact-note,.reader-fact-source{font-size:.85rem;line-height:1.5;margin-top:6px;color:var(--reader-muted)}
.reader-attention{padding:12px 16px;border-left:3px solid #ae7a39;background:#fff8ee;margin:18px 0}
.reader-attention h3{margin:0 0 6px}.reader-attention ul{padding-left:20px;margin:0}
.reader-actions{display:flex;align-items:center;flex-wrap:wrap;gap:14px 20px;margin-top:20px;font-size:.95rem}
.reader-actions .reader-action{background:var(--reader-accent);color:white;padding:9px 16px;border-radius:7px;text-decoration:none}
.reader-content details.reader-basics{padding:0;border:0;margin:14px 0;font-size:.9rem}
.reader-tools h2{margin:0;font-size:1.2rem}.reader-header{overflow-wrap:anywhere}
.reader-content .reader-evidence{padding:0;margin:12px 0;background:#fff}
.reader-evidence>summary{padding:16px 18px;margin:0 0 0 20px!important;line-height:1.45}
.reader-evidence>summary .reader-detail-hint{display:block;font-weight:400;font-size:.9rem;color:var(--reader-muted);margin-top:4px}
.reader-evidence>div{padding:4px 20px 18px;border-top:1px solid var(--reader-line)}
.reader-content .reader-nested{background:#f8faf8}.reader-nested>div{padding-top:8px}
.reader-content .reader-limit-detail{border:0;padding:0;margin:6px 0;background:transparent}
.reader-limit-detail>summary{font-weight:400!important;margin-left:0!important}.reader-limit-detail>div{padding:6px 0}
.reader-orientation{padding-bottom:18px;margin-bottom:22px}
.reader-facts{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.reader-fact{flex:1 1 180px;max-width:340px;padding:12px 16px}
.reader-fact-value{font-size:1.5rem}.reader-fact-label{margin-top:4px;font-size:.92rem}
.reader-fact-source{display:none}.reader-fact-note{margin-top:3px}
.reader-orientation .reader-basics{color:var(--reader-muted)}
.reader-limits ul{padding-left:20px}.reader-limits li+li{margin-top:8px}
.reader-section h2{scroll-margin-top:16px}
.reader-content .reader-reading-note{font-size:.9rem;color:var(--reader-muted);margin:0 0 14px}
.reader-candidate summary{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.reader-candidate summary>span{min-width:4em}.reader-candidate .reader-aso-sequence{display:block;margin-top:6px;font-size:.95rem;font-weight:400;overflow-wrap:anywhere;width:fit-content;max-width:100%}
.reader-candidate dl{margin:0}.reader-candidate dt{font-size:.85rem;color:var(--reader-muted);margin-top:10px}.reader-candidate dd{margin:0;overflow-wrap:anywhere}
.reader-candidate summary>.reader-candidate-check{flex-basis:100%;font-size:.85rem;font-weight:400;color:var(--reader-muted);line-height:1.5}
.reader-panel-chemistry{font-size:.92rem;margin:0 0 18px}.reader-panel-chemistry h3{margin:0 0 6px}.reader-panel-chemistry p{margin:6px 0}
@media(max-width:600px){.reader-facts{gap:8px}.reader-fact{padding:12px;flex-basis:calc(50% - 4px);max-width:none}.reader-fact-value{font-size:1.6rem}.reader-fact-label{font-size:.9rem}.reader-orientation h2{font-size:1.35rem}}
:focus-visible{outline:3px solid #23624e;outline-offset:3px}
@media(max-width:600px){.reader-page{padding:20px 12px 40px}.reader-content{padding:18px 14px}.reader-header h1{font-size:1.6rem}.reader-glossary dl{display:block}.reader-glossary dd{margin:0 0 14px}.reader-content details{padding:10px}}
@media print{body{background:white}.reader-page{max-width:none;padding:0}.reader-tools,.reader-nav{display:none}.reader-content{border:0;padding:0}.reader-content .table-scroll{overflow:visible}.reader-content table{width:100%;min-width:0}.reader-content td{overflow-wrap:anywhere}.reader-section{break-inside:avoid}}
"""

# These are presentation labels, not substitutions in generated assertions.
HEADING_LABELS = {
    "run output summary": "Results",
    "sandbox summary": "Analysis results",
    "general result": "Completion details",
    "main files to open": "Report files",
    "requested output files": "Download your files",
    "how to read the rankings": "How to read the rankings",
    "where users should look to choose asos": "Choosing candidates",
    "top normal-ranking candidates": "Ranked candidates",
    "selected physical panel": "Selected ASOs",
    "physical chemistry": "ASO chemistry",
    "selection and evidence": "How candidates were selected",
    "selected-panel off-target search budget": "How broadly off-target matches were checked",
    "predicted splice-response candidate sets": "Predicted splicing changes",
    "strongest raw splice-effect rows": "Largest predicted splicing changes",
    "requested binding-region candidate sets": "Candidates for the requested sites",
    "conditional exon-skipping sequence consequences": "Exon-skipping scenarios",
    "practical interpretation": "What the results mean",
    "outcome summary": "Results at a glance",
    "candidate sequence audit and reconciliation": "Candidate sequence checks",
    "full observation membership": "All comparison results",
    "complete audit table": "Sequence check details",
    "exclusion reasons": "Why candidates were excluded",
    "design run status": "Results by target",
    "design runs": "Design reports",
    "aso set comparisons": "Panel comparisons",
    "bottom line": "Main finding",
    "main uncertainty": "What remains uncertain",
    "caveats": "Limitations",
    "task outcomes": "Analysis steps",
    "detailed response": "Full analysis",
    "recorded panel comparison": "Panel comparison",
    "lead records (recorded rank 1)": "Top-ranked candidates in each panel",
    "oligoai gapmer efficacy prediction": "Predicted gapmer activity (OligoAI)",
    "molecule overview": "Supplied ASOs and sequence checks",
    "reference context": "Reference sequences used",
    "exact gene-screening count definition": "What the off-target count includes",
    "top allele-specific candidates": "Candidates for the requested variant",
    "target-rna accessibility and screening scope": "RNA accessibility and search coverage",
    "how to read this run": "Where to start",
    "what happened": "How the analysis was done",
    "branch outcomes": "Results by analysis type",
    "run metadata": "Run information",
    "how to read the outputs": "Understanding the output files",
    "aso-by-aso assessment": "Assessment of each ASO",
    "original request": "Your full request",
    "what needs attention": "What needs attention",
    "supporting results and methods": "Evidence and methods",
    "sequence orientation": "Which strand each sequence represents",
    "assumptions": "Assumptions to review",
    "suggested follow-up": "Suggested follow-up",
    "original intervals and assembly conversion evidence": "Requested regions and genome coordinates",
    "translation-start evidence for the selected panel": "Binding near the start codon",
    "recommended shortlist": "Selection checks for each ASO",
    "source search evidence": "Binding-site datasets",
    "protected-transcript sequence discrimination": "Matches to the isoform to preserve",
}

# Fold only explicitly named methods/administration, never arbitrary prose or
# an unfamiliar section that might contain an important result or caveat.
DETAIL_HEADINGS = {
    "method", "methods", "general result", "main files to open", "task outcomes",
    "how to read the rankings", "where users should look to choose asos",
    "references and search scope", "off-target search references",
    "selected-panel off-target search budget", "selection and evidence",
    "what happened", "key decisions", "how to read the outputs", "run metadata",
    "reference context", "exact gene-screening count definition", "glossary",
    "original request", "how to read this run", "original intervals and assembly conversion evidence",
    "source search evidence",
}
# These are complete evidence blocks, not alternative summaries. Their contents
# are never shortened, sampled, deduplicated, or interpreted by the renderer.
EVIDENCE_HEADINGS = {
    "selected physical panel": "Sequences and binding positions.",
    "physical chemistry": "Sugar modifications and backbone chemistry for each ASO.",
    "top normal-ranking candidates": "Search candidates in recorded rank order; this list can differ from the selected panel.",
    "top allele-specific candidates": "Matches to the mutant and reference sequences.",
    "predicted splice-response candidate sets": "Candidate groups for predicted splicing changes; the response site can differ from the binding site.",
    "strongest raw splice-effect rows": "A check of model signals, separate from the selected panel.",
    "requested binding-region candidate sets": "Candidates grouped by requested binding region; the same molecule may appear in several groups.",
    "conditional exon-skipping sequence consequences": "RNA sequence changes calculated for hypothetical exon skips.",
    "target-rna accessibility and screening scope": "Predicted RNA accessibility and the recorded scope of screening.",
    "oligoai gapmer efficacy prediction": "Activity predictions and unavailable results, with their reasons.",
    "full observation membership": "Every supplied entry and the measures used in the comparison.",
    "complete audit table": "Supplied sequence records and check outcomes.",
    "exclusion reasons": "Which stated rules each excluded sequence did not meet.",
    "molecule overview": "Supplied sequences and matching RNA sites.",
    "assessment": "The recorded interpretation for each supplied ASO.",
    "aso-by-aso assessment": "Open an ASO for its interpretation, uncertainty and supporting evidence.",
    "recorded panel comparison": "Compare each measure within its stated reference and search scope.",
    "lead records (recorded rank 1)": "The first-ranked entry in each panel.",
    "design runs": "The full findings for each target, including incomplete work.",
    "aso set comparisons": "Detailed comparisons of the supplied panels.",
    "detailed response": "The complete recorded analysis and supporting explanations.",
    "supporting results": "The recorded results and details for individual entries.",
    "supporting evidence": "The evidence supporting the finding above, with its stated scope.",
    "supporting results and methods": "The recorded results and how they were calculated.",
    "sequence orientation": "The supplied ASO strand and the complementary target RNA site are different sequences.",
    "assumptions": "Conditions and input interpretations used in this analysis.",
    "suggested follow-up": "The report's proposed next steps, with their original scope and qualifications.",
    "recommended shortlist": "Selection order, off-target checks and sequence features to review.",
    "translation-start evidence for the selected panel": "Where each ASO binds relative to the start of the coding sequence.",
    "protected-transcript sequence discrimination": "Sequence matches and mismatches in the transcript the design aims to preserve.",
}
RESULT_HEADINGS = {
    "outcome summary", "practical interpretation", "bottom line", "selected physical panel",
    "design run status", "recorded panel comparison", "lead records (recorded rank 1)", "main finding",
}
LIMIT_HEADINGS = {"caveats", "limitations", "main uncertainty", "what remains uncertain", "what needs attention"}

TABLE_LABELS = {
    'id': 'ID', 'set': 'Set', 'sequence': 'Sequence',
    'target_score': 'Target score', 'offtarget_hits': 'Off-target matches',
    'source_row': 'Input row', 'source_row_number': 'Input row',
    'pooled_nondominated_frontier': 'A trade-off option?',
    'dominated_by_ids': 'Better alternatives on the compared measures',
    'dominated_by_count': 'Number with better compared measures',
    'set_total_count': 'Candidates in set', 'set_frontier_count': 'Set members on the comparison frontier',
    'set_dominated_count': 'Set members dominated in the comparison', 'set_unknown_count': 'Unknown in set',
    'claimed_length': 'Reported length', 'claimed_gc': 'Reported GC fraction',
    'recomputed_length': 'Calculated length', 'gc_count': 'G and C bases',
    'recomputed_gc_fraction': 'Calculated GC fraction',
    'recomputed_gc_fraction_exact': 'Exact GC fraction',
    'recomputed_gc_fraction_3dp': 'GC fraction (3 decimals)',
    'longest_homopolymer': 'Longest repeated-base run',
    'longest_homopolymer_base': 'Repeated base',
    'claimed_length_accuracy': 'Reported length correct?',
    'claimed_gc_accuracy': 'Reported GC correct?',
    'claimed_gc_rounded_3dp': 'Reported GC (3 decimals)',
    'inaccurate_claim_flags': 'Incorrect reported values',
    'retention_status': 'Selection outcome', 'exclusion_reasons': 'Reasons for exclusion',
    'exclusion_reason_count': 'Number of reasons',
    'unverified_retention_criteria': 'Criteria not checked',
    'native rank / scope': 'Rank within design group',
    'native score': 'Recorded score',
    'claimed_length accuracy': 'Reported length correct?',
    'claimed_gc accuracy': 'Reported GC correct?',
    'retention': 'Meets the stated rules?',
    'exclusion reasons': 'Reasons for exclusion',
    'aso (5′–3′)': 'Recorded ASO / sequence role (5′→3′)',
    'antisense oligo (5′→3′)': 'ASO sequence (5′→3′)',
    'genomic screening': 'Genome search',
    'source sequence / role (5′–3′)': 'ASO and target strands (5′→3′)',
    'order': 'Panel order',
    'directional delta': 'Predicted change in the requested direction',
    'opposite delta': 'Predicted change in the opposite direction',
    'exact off-target count': 'Exact off-target matches',
    'exact antisense records': 'Recorded exact matches',
    'minimum retained antisense mismatches': 'Fewest base differences in retained matches',
    'other-gene screening count': 'Other genes matched',
    'route': 'Predicted splice-site response',
    'candidates': 'ASOs in this group',
    'confidence': 'Confidence in this approach',
}
# Redundant arithmetic and input-row bookkeeping can be opened on demand.
# Unknown columns and all decision/result/limitation columns remain visible.
DETAIL_COLUMNS = {
    'source_row', 'source_row_number', 'claimed_length', 'claimed_gc', 'gc_count',
    'recomputed_gc_fraction_exact', 'recomputed_gc_fraction_3dp',
    'longest_homopolymer_base', 'claimed_gc_rounded_3dp', 'exclusion_reason_count',
    'set_total_count', 'set_frontier_count', 'set_dominated_count', 'set_unknown_count',
    'native rank / scope', 'physical binding interval(s), bed0', 'target genomic strand',
    'physical splice element', 'predicted response site', 'reference tm (°c)',
    'reference δg (kcal/mol)',
}

STATUS_LABELS = {
    'completed_with_candidates': 'Candidates produced',
    'completed_no_candidates': 'No candidates produced',
    'outputs_present': 'Output files available',
    'not_predicted': 'Prediction unavailable',
    'not_applicable': 'Not applicable',
    'inference_error': 'Prediction failed',
    'not_requested': 'Outside this analysis',
    'incomplete_or_unavailable': 'Incomplete',
    'complete_within_declared_scope': 'Completed',
    'provisional': 'Some evidence unavailable',
    'computational_candidate': 'Computational checks available',
    'exploratory': 'Prediction requires review',
    'not attested': 'Not verified',
    'not_screened': 'Search not performed',
    'complete': 'Complete',
    'unmeasured': 'Measurements unavailable',
    'unavailable': 'Unavailable',
    'retained': 'Kept',
    'excluded': 'Excluded',
    'indeterminate': 'Unresolved',
}

# These labels apply only to the corresponding scientific field. An identifier
# or arbitrary prose containing the same text is never rewritten.
MISSING_RESULT_COLUMNS = {
    'exact off-target count', 'off-target genes', 'off-target matches', 'other-gene screening count',
    'offtarget_hits', 'predicted inhibition (%)', 'minimum retained antisense mismatches',
    'coverage fraction', 'reference tm (°c)', 'reference δg (kcal/mol)',
}
from .report_candidate_view import MOTIF_LABELS


def _reason_label(value):
    """Decode explicit conditions, retaining arbitrary thresholds exactly."""
    if value == 'sequence_unavailable':
        return 'Sequence missing'
    patterns = (
        (r'gc_fraction_outside_([0-9.]+)_to_([0-9.]+)', lambda m: f'GC fraction outside {m[1]}–{m[2]}'),
        (r'length_not_exactly_(\d+)_nt', lambda m: f'Length differs from {m[1]} bases'),
        (r'longest_homopolymer_gt_(\d+)', lambda m: f'Repeated-base run longer than {m[1]} bases'),
    )
    for pattern, render in patterns:
        match = re.fullmatch(pattern, value)
        if match:
            return render(match)
    return value


def friendly_tables(fragment):
    def values(table_match):
        table = table_match[0]
        headers = [html.unescape(re.sub('<[^>]+>', '', cell)).casefold() for cell in re.findall(r'<th>(.*?)</th>', table, re.S)]
        def row(row_match):
            index = -1
            def cell(cell_match):
                nonlocal index
                index += 1
                key = headers[index] if index < len(headers) else ''
                raw = html.unescape(re.sub('<[^>]+>', '', cell_match[1]))
                if key in {'antisense oligo (5′→3′)', 'source sequence / role (5′–3′)', 'aso (5′–3′)'}:
                    strands = re.fullmatch(r'([ACGTUacgtu]+) \(transcript_oriented_target_binding_site\); synthesized ASO: ([ACGTUacgtu]+|unverified)', raw)
                    if strands:
                        # Both strings and their roles are explicitly recorded.
                        # Never calculate a complement or guess an orderable ASO.
                        aso = ('<span>Recorded ASO:</span> <code>' + html.escape(strands[2]) + '</code>'
                               if strands[2] != 'unverified' else '<span>ASO strand not verified</span>')
                        return ('<td data-recorded-value="' + html.escape(raw, quote=True) + '">'
                                + aso + '<details class="reader-strands"><summary>Target RNA strand</summary>'
                                + '<p>Target RNA site (5′→3′): <code>' + html.escape(strands[1]) + '</code></p>'
                                + '<p>This is the RNA sequence the ASO is designed to pair with. Sugar and backbone modifications are listed under ASO chemistry.</p></details></td>')
                if key == 'native rank / scope' and (rank := re.fullmatch(r'(\d+|unavailable|not recorded)\s+/\s+(.+)', raw)):
                    rank_label = rank[1] if rank[1].isdigit() else 'No rank supplied'
                    return ('<td data-recorded-value="' + html.escape(raw, quote=True)
                            + '"><details class="reader-rank"><summary>' + rank_label
                            + ' · group details</summary><p>Design group: <code>'
                            + html.escape(rank[2]) + '</code></p></details></td>')
                if key == 'genomic screening':
                    from .report_candidate_view import screening_fields, screening_summary
                    if fields := screening_fields(raw):
                        return ('<td data-recorded-value="' + html.escape(raw, quote=True) + '">'
                                + '<details><summary>' + html.escape(screening_summary(raw)) + '</summary>'
                                + value_html(dict(fields)) + '</details></td>')
                if key in {'status', 'outcome', 'retention_status', 'off-target screen', 'splice safeguards', 'screening status'}:
                    label = ({'passed': 'Within the specified limits', 'failed': 'Outside the specified limits'}[raw]
                             if key == 'splice safeguards' and raw in {'passed', 'failed'}
                             else STATUS_LABELS.get(raw, VALUE_LABELS.get(raw, raw)))
                elif key in MISSING_RESULT_COLUMNS and raw in {'None', 'null', 'NA', 'N/A', 'unknown'}:
                    label = 'Not reported'
                elif key == 'motif flags':
                    try:
                        flags = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        flags = None
                    label = ('; '.join(MOTIF_LABELS.get(flag, flag) for flag in flags) or 'None flagged') if (
                        isinstance(flags, list) and all(isinstance(flag, str) for flag in flags)) else raw
                elif key in {'exclusion reason', 'exclusion reasons', 'exclusion_reasons'}:
                    label = _reason_label(raw)
                elif key in {'pooled_nondominated_frontier', 'claimed_length_accuracy', 'claimed_gc_accuracy'} and raw in {'true', 'false', 'True', 'False'}:
                    label = 'Yes' if raw.lower() == 'true' else 'No'
                elif key == 'reference / context':
                    label = re.sub(r'\b(genomic_pre_mrna|mature_mrna|mature_rna)\b', lambda m: VALUE_LABELS[m[0]], raw)
                elif not identity_key(key):
                    label = VALUE_LABELS.get(raw, raw)
                else:
                    label = raw
                if key != 'motif flags' and (record := parse_collection(raw)) is not None and not identity_key(key):
                    return ('<td data-recorded-value="' + html.escape(raw, quote=True) + '">'
                            + value_html(record, key) + '</td>')
                if label == raw:
                    return cell_match[0]
                # The source code remains machine-checkable and available in the
                # tooltip and unchanged text report. Numeric cells never change.
                return '<td data-recorded-value="' + html.escape(raw, quote=True) + '"><abbr title="Recorded value: ' + html.escape(raw, quote=True) + '">' + html.escape(label) + '</abbr></td>'
            return re.sub(r'<td>(.*?)</td>', cell, row_match[0], flags=re.S)
        return re.sub(r'<tr>.*?</tr>', row, table, flags=re.S)
    fragment = re.sub(r'<table>.*?</table>', values, fragment, flags=re.S)
    def header(match):
        # Writers sometimes use inline code around a known field name. Strip
        # only that formatting, never links or arbitrary header markup.
        value = html.unescape(re.sub(r'</?code>', '', match[1])); key = value.strip('`').casefold()
        if key not in TABLE_LABELS and key not in DETAIL_COLUMNS and key not in FIELD_LABELS and not re.fullmatch(r'[A-Za-z][\w]*_[\w]+', value):
            return match[0]
        css = ' class="reader-detail-column"' if key in DETAIL_COLUMNS else ''
        return '<th' + css + ' title="' + html.escape(value, quote=True) + '">' + html.escape(TABLE_LABELS.get(key, field_label(value))) + '</th>'
    return re.sub(r'<th>(.*?)</th>', header, fragment, flags=re.S)

TERMS = (
    (r"\bASOs?\b|antisense", "ASO", "An antisense oligonucleotide: a short synthetic strand designed to bind RNA."),
    (r"\bloci\b|\blocus\b", "Locus / loci", "A locus is one position in a reference genome. Loci are counted as positions; several can belong to the same gene."),
    (r"\bPMO\b|morpholino", "PMO", "Phosphorodiamidate morpholino oligomer: an ASO with morpholino units and phosphorodiamidate links."),
    (r"safeguard", "Splice safeguard", "A check that predicted changes at a specified splice site stay within the requested limits."),
    (r"\bRNA\b|transcript", "RNA / transcript", "RNA carries or helps use genetic instructions. A transcript is one RNA product of a gene; different versions can contain different regions."),
    (r"splic", "Splicing", "How pieces of RNA are joined. Changes in splicing can change which RNA version is produced."),
    (r"gapmer", "Gapmer", "An ASO with a central DNA segment and modified ends, designed to recruit RNase H to cut the target RNA."),
    (r"steric", "Steric blocker", "An ASO designed to occupy an RNA site and block a process such as splicing or translation."),
    (r"phosphorothioate|\bPS\b", "Phosphorothioate (PS)", "A chemical modification to the links between nucleotides."),
    (r"MOE|2.prime", "2′-MOE", "A sugar modification used in some ASOs. Its placement is part of the chemistry record, separate from the A, C, G and T sequence letters."),
    (r"lna_383|\bLNA\b", "LNA / 3–8–3", "LNA is a sugar modification. A 3–8–3 gapmer has three LNA nucleotides at each end and eight DNA nucleotides in the middle. A motif flag naming that study identifies the chemistry studied."),
    (r"cpg_immune_context|\bCpG\b", "CpG motif", "A C followed by a G in the ASO sequence. Its biological significance depends on chemistry, methylation, sequence and species."),
    (r"\bdelta\b", "Predicted change (delta)", "The change in a model score. For splicing, the score describes use of a particular splice site."),
    (r"BED0|0.based|1.based|coordinate", "Sequence coordinates", "Positions depend on the named reference and counting convention. Zero-based half-open intervals exclude the end; one-based inclusive intervals include both endpoints. RNA and genomic positions are not interchangeable."),
    (r"predict|estimate", "Prediction", "A result calculated by a model."),
    (r"\bisoform\w*\b", "Isoform", "A version of an RNA transcript."),
    (r"off[-_ ]target", "Off-target", "Another sequence an ASO may also bind. The search settings specify which references and sequence differences were checked."),
    (r"\bGC(?:_|\b)", "GC fraction", "The fraction of a sequence made up of G and C bases. A fraction of 0.30 is 30%."),
    (r"homopolymer", "Homopolymer", "A run of the same base, such as AAAA."),
    (r"nondominated|pareto", "Nondominated candidate", "An option for which improving one compared measure requires a trade-off on another."),
    (r"\brank\b|selection_rank", "Rank", "The order recorded for a particular candidate set. Ranks from different sets or scoring methods may not be comparable."),
    (r"\b(?:unavailable|not_predicted|not recorded)\b", "Unavailable result", "A missing measurement or prediction; the accompanying note explains why."),
    (r"\bN/?A\b", "NA / N/A", "A field with no value supplied. The accompanying note explains whether the value is missing or inapplicable."),
)


def glossary_html(text):
    terms = [(name, meaning) for pattern, name, meaning in TERMS if re.search(pattern, text, re.I)]
    if not terms:
        return ""
    return ('<details class="reader-glossary" id="reader-terms"><summary>Terms used in this report</summary><dl>'
            + ''.join('<dt>' + html.escape(name) + '</dt><dd>' + html.escape(meaning) + '</dd>' for name, meaning in terms)
            + '</dl></details>')


def _sections(markdown):
    """Split top-level report sections without interpreting fenced content."""
    sections, current, fence = [], {"heading": "", "level": 0, "lines": []}, None
    for line in str(markdown or '').splitlines():
        stripped = line.strip()
        marker = re.match(r'^(`{3,}|~{3,})', stripped)
        if fence:
            current['lines'].append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            continue
        if marker:
            fence = marker[1]
        heading = re.match(r'^(#{1,2})\s+(.+?)\s*$', line) if not fence else None
        if heading:
            if current['heading'] or current['lines']:
                sections.append(current)
            current = {"heading": heading[2], "level": len(heading[1]), "lines": []}
        else:
            current['lines'].append(line)
    if current['heading'] or current['lines']:
        sections.append(current)
    return sections


def _source_search_completed(lines):
    """Fold verified source-search provenance; keep missing/failed searches open."""
    blocks = re.findall(r'```(?:json)?\s*\n(.*?)\n```', '\n'.join(lines), re.S)
    records = [parse_collection(block) for block in blocks]
    return bool(records) and all(isinstance(record, dict)
        and record.get('search_status') in ('complete', 'completed') for record in records)


def _intro_html(lines, report_dir, extra_details, header_request):
    """Fold only the generated request/administration blocks; keep their text."""
    # Only a leading metadata block is eligible. The same text inside a code
    # example or after scientific prose must stay in its original context.
    file_indices = set()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if re.fullmatch(r'\s*(?:Input|Audit table|Output file):\s*`[^`]+`\s*\.?\s*', line):
            file_indices.add(index)
        else:
            break
    file_lines = [line for index, line in enumerate(lines) if index in file_indices]
    if file_lines:
        extra_details.append('<details><summary>Input and output files</summary>' + markdown_html('\n'.join(file_lines), report_dir) + '</details>')
        lines = [line for index, line in enumerate(lines) if index not in file_indices]
    text = '\n'.join(lines)
    before, marker, request = text.partition('\nOriginal request:\n')
    if marker:
        if request.strip() != str(header_request or '').strip():
            extra_details.append('<details><summary>Your request</summary>' + markdown_html(request, report_dir) + '</details>')
        return _intro_html(before.splitlines(), report_dir, extra_details, header_request)
    # Fold only the single administrative line. Any following scientific
    # context or qualification remains visible, even in the same paragraph.
    if text.lstrip().startswith('Output folder:') or re.match(r'\s*Integrated \d+ design result\(s\),', text):
        first, _, rest = text.lstrip().partition('\n')
        extra_details.append('<details><summary>Processing details</summary>' + markdown_html(first, report_dir) + '</details>')
        return markdown_html(rest, report_dir)
    return markdown_html(text, report_dir)


def _evidence_body(lines, report_dir):
    """Add a second reading level inside known evidence sections.

    A heading in a code fence is data. Unfamiliar top-level sections do not use
    this function and remain visible. Each original body is rendered once.
    """
    groups, current, fence = [], {'heading': '', 'lines': []}, None
    for line in lines:
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if fence:
            current['lines'].append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            continue
        if marker:
            fence = marker[1]
        heading = re.match(r'^###\s+(.+?)\s*#*\s*$', line) if not fence else None
        if heading:
            groups.append(current)
            current = {'heading': heading[1], 'lines': []}
        else:
            current['lines'].append(line)
    groups.append(current)
    parts = []
    for group in groups:
        body = friendly_tables(markdown_html('\n'.join(group['lines']), report_dir))
        if group['heading']:
            parts.append('<details class="reader-nested"><summary>' + html.escape(group['heading'])
                         + '</summary><div>' + body + '</div></details>')
        else:
            parts.append(body)
    return ''.join(parts)


def _disclosure(section_id, label, body, hint='', *, opened=False):
    return ('<details class="reader-evidence" id="' + section_id + '"' + (' open' if opened else '') + '><summary>' + html.escape(label)
            + ('<span class="reader-detail-hint">' + html.escape(hint) + '</span>' if hint else '')
            + '</summary><div>' + body + '</div></details>')


def _limitation_body(lines, report_dir):
    """Keep each caveat's first sentence visible; expand its exact explanation.

    Only simple, complete bullet lists qualify. Complex Markdown stays intact.
    Do not split inside a code span, link, or parenthesis, or invent a headline.
    """
    nonempty = [line for line in lines if line.strip()]
    if not nonempty or any(not line.startswith('- ') for line in nonempty):
        return friendly_tables(markdown_html('\n'.join(lines), report_dir))
    items = []
    for line in nonempty:
        text = line[2:]; end = None; code = False; brackets = 0
        # Native multi-observation caveats start with a scope label. Keep that
        # label with the consequence; a label alone is not a useful preview.
        scope = re.match(r'^\*\*Observation \d+ \(.+?\):\*\*\s+', text)
        consequence_start = scope.end() if scope else 0
        for index, char in enumerate(text):
            if char == '`':
                code = not code
            if code:
                continue
            if char in '[(':
                brackets += 1
            elif char in '])':
                brackets = max(0, brackets - 1)
            if (char == ':' and index > consequence_start and brackets == 0
                    and text[index + 1:index + 2].isspace()
                    and len(text[consequence_start:index].split()) <= 12):
                end = index + 1
                break
            if char == '.' and brackets == 0 and re.match(r'\s+[A-Z]', text[index + 1:]):
                end = index + 1
                break
        if end and len(text) > 180:
            first = markdown_html(text[:end], report_dir).strip().removeprefix('<p>').removesuffix('</p>')
            rest = markdown_html(text[end:].lstrip(), report_dir)
            items.append('<li><details class="reader-limit-detail"><summary>' + first + '</summary><div>' + rest + '</div></details></li>')
        else:
            items.append(markdown_html(line, report_dir).strip().removeprefix('<ul>').removesuffix('</ul>').strip())
    return '<ul>' + ''.join(items) + '</ul>'


def _group_design_evidence(evidence, sections):
    """Keep one path to the panel, its rationale, and the wider search."""
    if not any(sections[item[0]]['heading'].casefold() == 'selected physical panel' for item in evidence):
        return evidence
    rationale = {'practical interpretation', 'physical chemistry', 'recommended shortlist',
                 'predicted splice-response candidate sets', 'oligoai gapmer efficacy prediction',
                 'translation-start evidence for the selected panel', 'protected-transcript sequence discrimination'}
    alternatives = {'top normal-ranking candidates', 'top allele-specific candidates',
                    'strongest raw splice-effect rows', 'requested binding-region candidate sets'}
    primary, reasons, other = [], [], []
    for item in evidence:
        key = sections[item[0]]['heading'].casefold()
        (reasons if key in rationale else other if key in alternatives else primary).append(item)
    for group, anchor, label, hint in (
        (reasons, 'reader-rationale', 'Predictions and selection checks', 'Model results, selection reasons and checks on each ASO.'),
        (other, 'reader-other-designs', 'Explore the wider search', 'Additional candidate groups and their original rankings.'),
    ):
        if group:
            primary.append((-1, anchor, label, _disclosure(anchor, label, ''.join(item[3] for item in group), hint)))
    return primary


def readable_markdown_html(markdown, report_dir=None, *, include_tools=True, header_title=None, header_request=None, context=None, inventory_html='', downloads_html='', explorer=''):
    """Reorganize saved text for reading without editing its evidence blocks.

    Reordering is limited to known result/limitation sections. Every original
    body is rendered once. Unknown headings are visible in their original order.
    """
    sections = _sections(markdown)
    from .report_orientation import build_orientation, orientation_html
    orientation_data = build_orientation(sections, context)
    visible, details, evidence, intro, downloads = [], [], [], [], []
    panel_indices = [i for i, s in enumerate(sections) if s['heading'].casefold() == 'selected physical panel']
    chemistry_indices = [i for i, s in enumerate(sections) if s['heading'].casefold() == 'physical chemistry']
    inline_chemistry = chemistry_indices[0] if len(panel_indices) == len(chemistry_indices) == 1 else None
    for index, section in enumerate(sections):
        if index == inline_chemistry:
            continue
        heading = section['heading']; key = heading.casefold()
        label = HEADING_LABELS.get(key, heading)
        body = friendly_tables(markdown_html('\n'.join(section['lines']), report_dir))
        section_id = f'reader-section-{index}'
        if not heading or section['level'] == 1:
            if key == 'supplied aso evaluation':
                # Native multi-ASO reports retain complete, reviewed summaries
                # under observation headings. Each can be opened independently.
                content = _evidence_body(section['lines'], report_dir)
            else:
                content = friendly_tables(_intro_html(section['lines'], report_dir, details, header_request))
            # Generated design context is available beside the full request.
            # Free-form introductions may contain the only finding and stay open.
            if key == 'run output summary' and re.search(r'^This run designed ', '\n'.join(section['lines']), re.M):
                details.append(_disclosure(section_id, 'Design context and assumptions', content))
                continue
            if not content.strip():
                continue
            if heading and label != header_title and key not in {'run output summary', 'sandbox summary'}:
                content = '<h2>' + html.escape(label) + '</h2>' + content
            intro.append((index, section_id, label or 'Overview', content))
        elif (key == 'outcome summary' and orientation_data['kind'] == 'sequence_checks' and orientation_data['facts']
              and all(not line.strip() or re.fullmatch(r'\s*-\s*(Total input rows|Retained|Excluded|Indeterminate):\s*\d+\s*', line)
                      for line in section['lines'])):
            details.append(_disclosure(section_id, 'Sequence totals', body))
        elif key == 'exclusion reasons' and orientation_data.get('title') == 'No ASOs selected':
            visible.append((0, (index, section_id, label,
                '<section class="reader-section" id="' + section_id + '"><h2>' + html.escape(label) + '</h2>' + body + '</section>')))
        elif key == 'practical interpretation' and (context or {}).get('kind') == 'design':
            evidence.append((index, section_id, label, _disclosure(section_id, label,
                _evidence_body(section['lines'], report_dir))))
        elif key in EVIDENCE_HEADINGS:
            lines = section['lines']
            counts = [line for line in lines if key == 'selected physical panel' and
                      re.fullmatch(r'Selected candidate rows: \d+\. Distinct molecules \(antisense sequence plus chemistry\): \d+\. Preview: \d+ molecules in .+\.', line)]
            content = _evidence_body([line for line in lines if line not in counts], report_dir)
            if counts:
                content += '<details><summary>Selection counts</summary>' + markdown_html('\n'.join(counts), report_dir) + '</details>'
            if key == 'selected physical panel':
                from .report_candidate_view import candidate_reading_view, panel_check_records
                from .report_orientation import _candidate_summary, section_tables
                reading_view = candidate_reading_view(section, checks=panel_check_records(section, sections))
                if reading_view:
                    counts_view = _candidate_summary([section], [(0, table) for table in section_tables(section)])
                    preview = counts_view and counts_view['shown'] < counts_view['distinct']
                    table_label = 'Preview table' if preview else 'Full selected-ASO table'
                    content = reading_view + '<details><summary>' + table_label + '</summary>' + content + '</details>'
                    if preview and inventory_html:
                        content += '<p><a href="#reader-design-inventory">Browse all design records</a></p>'
                if inline_chemistry is not None:
                    chemistry_lines = sections[inline_chemistry]['lines']
                    chemistry = _evidence_body(chemistry_lines, report_dir)
                    # Keep the concise physical identity beside the sequences;
                    # remaining chemistry paragraphs retain their own disclosure.
                    first, closing, rest = chemistry.partition('</p>')
                    specifications = sum(line.strip().startswith('Sugars (5′→3′):') for line in chemistry_lines)
                    if specifications > 1:
                        chemistry = ('<p>This panel includes multiple chemistry specifications.</p>'
                                     '<details><summary>View the chemistry specifications</summary>' + chemistry + '</details>')
                    elif specifications == 1 and closing and first.lstrip().startswith('<p>'):
                        chemistry = first + closing + ('<details><summary>Chemistry details</summary>' + rest + '</details>' if rest.strip() else '')
                    content = ('<section class="reader-panel-chemistry" id="reader-section-' + str(inline_chemistry)
                               + '"><h3>ASO chemistry</h3>' + chemistry + '</section>' + content)
            elif key == 'oligoai gapmer efficacy prediction':
                from .report_candidate_view import prediction_reading_view
                reading_view = prediction_reading_view(section)
                if reading_view:
                    content = reading_view + '<details><summary>All prediction records and model details</summary>' + content + '</details>'
            elif key == 'predicted splice-response candidate sets':
                content = ('<p>Acceptor sites mark the start of an exon; donor sites mark its end. '
                           'Each row groups ASOs by their predicted effect on one splice site. '
                           'The changes are unitless model scores. Confidence describes the ranking of these approaches.</p>' + content)
            disclosure = _disclosure(section_id, label, content, EVIDENCE_HEADINGS[key],
                                     opened=key == 'selected physical panel')
            if key == 'conditional exon-skipping sequence consequences':
                details.append(disclosure)
            else:
                evidence.append((index, section_id, label, disclosure))
        elif key in DETAIL_HEADINGS and (key != 'source search evidence' or _source_search_completed(section['lines'])):
            details.append('<details id="' + section_id + '"><summary>' + html.escape(label)
                           + '</summary><div>' + body + '</div></details>')
        else:
            cls = ' reader-limits' if key in LIMIT_HEADINGS else ' reader-downloads' if key == 'requested output files' else ''
            if key in LIMIT_HEADINGS:
                body = _limitation_body(section['lines'], report_dir)
            elif key in {'practical interpretation', 'main finding', 'bottom line'}:
                body = _evidence_body(section['lines'], report_dir)
            elif key == 'requested output files':
                body = body.replace('<a ', '<a data-download-priority="primary" ')
            content = '<h2>' + html.escape(label) + '</h2>' + body
            item = (index, section_id, label, '<section class="reader-section' + cls + '" id="' + section_id + '">' + content + '</section>')
            if key == 'requested output files':
                downloads.append(item)
            else:
                visible.append((0 if key in RESULT_HEADINGS else 1 if key in LIMIT_HEADINGS else 2, item))
    visible.sort(key=lambda entry: (entry[0], 0 if entry[1][2] == 'Main finding' else 1, entry[1][0]))
    evidence.sort(key=lambda item: (0 if sections[item[0]]['heading'].casefold() in
                                   {'selected physical panel', 'molecule overview', 'aso-by-aso assessment'} else 1, item[0]))
    if (context or {}).get('kind') == 'design':
        evidence = _group_design_evidence(evidence, sections)
    if inventory_html:
        evidence.append((-1, 'reader-design-inventory', 'All design records',
            _disclosure('reader-design-inventory', 'All design records', friendly_tables(inventory_html),
                        'Searchable tables with every saved design entry. Each table states its scope.')))
    blocks = [item for _, item in visible]
    limits = next((item for priority, item in visible if priority == 1), None)
    if limits and orientation_data['notices']:
        from .report_orientation import notice_list_html
        # One limitations section contains both the source's explanation and
        # the checked status summaries. All source paragraphs remain intact.
        merged = (*limits[:3], limits[3].replace('</h2>', '</h2>' + notice_list_html(orientation_data), 1))
        blocks = [merged if item == limits else item for item in blocks]
        limits = merged
        orientation_data = {**orientation_data, 'notices': []}
    terms = glossary_html(str(markdown or ''))
    tools = ('<div class="reader-tools"><span id="reader-findings"></span>'
             '<button type="button" class="reader-expand">Expand all details</button></div>') if include_tools else '<span id="reader-findings"></span>'
    # A free-form introduction may itself be the only scientific summary.
    lead = ''.join('<section class="reader-section" id="' + item[1] + '">' + item[3] + '</section>' for item in intro)
    detail_html = (_disclosure('reader-details', 'Methods, references and run details', ''.join(details),
                              'How the analysis was done, which references were used, and the complete run context.')
                   if details else '')
    if downloads and downloads_html:
        # Existing requested deliverables and supplemental output links share
        # one destination. Retain the original requested-files source anchor.
        first = downloads[0]
        before, closing, after = first[3].rpartition('</section>')
        downloads_html = downloads_html.replace('<h2>Download files</h2>', '<h3>Additional output files</h3>', 1)
        downloads[0] = (*first[:3], before + downloads_html + closing + after)
        downloads_html = ''
    main_findings = [item for item in blocks if sections[item[0]]['heading'].casefold() in {'main finding', 'bottom line'}]
    compact = orientation_data['kind'] != 'design' and bool(main_findings)
    orientation = orientation_html(orientation_data, findings='reader-findings',
                                   downloads=downloads[0][1] if downloads else 'reader-output-files' if downloads_html else None,
                                   limitations=limits[1] if limits else None, compact=compact)
    # A specific finding already introduces an unstructured analysis. Do not
    # put a second, generic "this is an analysis" preamble in front of it.
    if (orientation_data['kind'] == 'analysis' and not orientation_data['facts']
            and not orientation_data['notices']
            and any(section['heading'].casefold() in {'main finding', 'bottom line'} for section in sections)):
        orientation = ''
    if compact:
        opening = lead + ''.join(item[3] for item in main_findings) + orientation
        blocks = [item for item in blocks if item not in main_findings]
    else:
        opening = orientation + lead
    if explorer:
        # The explorer supplies the opening facts and candidate view. Preserve
        # every source paragraph and stable anchor in the deeper analysis.
        from .report_orientation import notice_list_html
        opening = notice_list_html(orientation_data)
        analysis = lead + ''.join(item[3] for item in blocks) + ''.join(item[3] for item in evidence)
        blocks = []
        evidence = [(-1, 'reader-analysis', 'Analysis and selection evidence',
                     _disclosure('reader-analysis', 'Analysis and selection evidence', analysis))] if analysis.strip() else []
    from .report_tables import page_large_tables
    return explorer + page_large_tables(readable_prose(opening + ''.join(item[3] for item in blocks)
            + tools
            + ''.join(item[3] for item in evidence) + ''.join(item[3] for item in downloads) + downloads_html
            + detail_html + terms))


READER_SCRIPT = """<script>
// In-page links can point to a folded explanation or a nested source section.
function revealReaderAnchor(hash=location.hash){
  let anchor=hash.slice(1);try{anchor=decodeURIComponent(anchor);}catch(_error){}
  if(anchor && !document.getElementById(anchor))window.revealReaderTableAnchor?.(anchor);
  if(anchor && !document.getElementById(anchor))window.prepareReportAnchor?.(anchor);
  const target=document.getElementById(anchor);
  if(!target)return;
  const panel=target.closest('.tab-panel');
  if(panel && !panel.classList.contains('active')){
    const tab=[...document.querySelectorAll('.tab-btn')].find(button=>button.dataset.tab===panel.id);
    if(tab)tab.click();
  }
  for(let node=target;node;node=node.parentElement){if(node.tagName==='DETAILS')node.open=true;}
  return target;
}
addEventListener('hashchange',()=>revealReaderAnchor());revealReaderAnchor();
// Clicking the same link again must reopen a section the reader has closed.
document.addEventListener('click',event=>{
  const link=event.target.closest('a[href^="#"]');
  if(!link||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey||event.button!==0)return;
  const target=revealReaderAnchor(link.hash);
  if(target)requestAnimationFrame(()=>{
    if(link.matches('.report-jump,.reader-skip'))target.focus({preventScroll:true});
    // Align the whole desktop exploration area when its map is sticky.
    // A sticky element's current position changes with the previous scroll.
    const scrollTarget=target.matches('.report-binding-view')&&getComputedStyle(target).position==='sticky'
      ?target.closest('.report-exploration-layout'):target;
    scrollTarget.scrollIntoView({block:'start'});
  });
});
document.querySelectorAll('.reader-content table').forEach(table=>{
  const columns=[...table.querySelectorAll('thead th')].map((th,index)=>th.classList.contains('reader-detail-column')?index:-1).filter(index=>index>=0);
  if(!columns.length)return;
  [...table.tBodies].forEach(body=>[...body.rows].forEach(row=>columns.forEach(index=>row.cells[index]?.classList.add('reader-detail-column'))));
  table.classList.add('reader-table-compact');
  const button=document.createElement('button');button.type='button';button.className='reader-column-toggle';
  button.textContent='Show all columns';button.setAttribute('aria-expanded','false');
  button.addEventListener('click',()=>{const compact=table.classList.toggle('reader-table-compact');button.textContent=compact?'Show all columns':'Show fewer columns';button.setAttribute('aria-expanded',String(!compact));});
  table.closest('.table-scroll').before(button);
});
document.querySelectorAll('.reader-expand').forEach(button=>button.addEventListener('click',()=>{
  const root=button.closest('.reader-content')||document;
  const opening=button.dataset.expanded!=='true';
  root.querySelectorAll('details').forEach(item=>{item.open=opening;});
  button.dataset.expanded=String(opening);
  button.textContent=opening?'Collapse details':'Expand all details';
}));
// Printing includes collapsed evidence without changing the on-screen state.
let readerPrintState=[],readerPrintTables=[];
addEventListener('beforeprint',()=>{readerPrintState=[...document.querySelectorAll('details')].map(item=>[item,item.open]);readerPrintState.forEach(([item])=>item.open=true);readerPrintTables=[...document.querySelectorAll('.reader-table-compact')];readerPrintTables.forEach(table=>table.classList.remove('reader-table-compact'));});
addEventListener('afterprint',()=>{readerPrintState.forEach(([item,open])=>item.open=open);readerPrintState=[];readerPrintTables.forEach(table=>table.classList.add('reader-table-compact'));readerPrintTables=[];});
</script><noscript><style>.reader-tools button,.reader-column-toggle,.tabs{display:none}</style></noscript>"""

from .report_tables import TABLE_CSS, TABLE_SCRIPT
from .report_theme import REPORT_CSS
from .report_explorer import EXPLORER_SCRIPT
READER_CSS += TABLE_CSS + REPORT_CSS
READER_SCRIPT = TABLE_SCRIPT + READER_SCRIPT + EXPLORER_SCRIPT


def concise_title(title, *, fallback='ASO analysis'):
    value = str(title or '').strip()
    if not value or len(value) > 100 or '\n' in value:
        return fallback
    return HEADING_LABELS.get(value.casefold(), value)


def basic_document(title, markdown, report_dir=None, *, candidate_data=None):
    display_title = concise_title(title)
    if display_title == 'ASO analysis':
        sections = _sections(markdown)
        first = next((section['heading'] for section in sections if section['level'] == 1), '')
        if first.casefold() in HEADING_LABELS and first.casefold() not in {'run output summary', 'sandbox summary'}:
            display_title = HEADING_LABELS[first.casefold()]
        if display_title == 'ASO analysis':
            from .report_orientation import build_orientation
            kind = build_orientation(sections)['kind']
            display_title = {'evaluation': 'Supplied ASO evaluation', 'comparison': 'ASO panel comparison',
                             'sequence_checks': 'Candidate sequence checks'}.get(kind, display_title)
    from .report_explorer import explorer_html
    if candidate_data is None and report_dir:
        from .report_explorer_data import project_analysis
        candidate_data = project_analysis(report_dir)
    body = readable_markdown_html(markdown, report_dir, header_title=display_title,
                                  explorer=explorer_html(candidate_data))
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            '<title>' + html.escape(display_title) + '</title><style>' + READER_CSS + '</style></head><body>'
            '<a class="reader-skip" href="#reader-main">Skip to results</a><div class="reader-page"><header class="reader-header"><p class="reader-brand">MODAL-ASO</p><h1>'
            + html.escape(display_title) + '</h1>'
            + ('<details><summary>Full report title</summary><p>' + html.escape(str(title)) + '</p></details>' if display_title != title and len(str(title or '')) > 100 else '')
            + '</header><main class="reader-content report-content" id="reader-main" tabindex="-1">' + body + '</main></div>' + READER_SCRIPT + '</body></html>')


def design_downloads(payload):
    """Keep local output links usable before JavaScript, including portable HTML."""
    from urllib.parse import urlsplit
    labels = {'Native candidate panel (CSV)': 'Design ASOs and chemistry (CSV)',
              'Complete native candidate evidence': 'All candidate details (JSON)',
              'Scored steric pool (Excel)': 'Scored steric candidates (Excel)',
              'Scored gapmer pool (Excel)': 'Scored gapmer candidates (Excel)'}
    links, secondary, seen, contents = [], [], set(), set()
    directory = payload.get('oligoai_report_dir') or (payload.get('meta') or {}).get('run_dir')
    artifacts = (payload.get('overview') or {}).get('primary_artifacts') or []
    inventory = (payload.get('outputs') or {}).get('artifacts') or {}
    if inventory.get('items'):
        from .html_report import _select_primary_artifacts
        artifacts = _select_primary_artifacts(inventory)
    for item in artifacts:
        href = item.get('href')
        if not isinstance(href, str) or not href or href in seen or urlsplit(href).scheme not in {'', 'file', 'http', 'https'}:
            continue
        seen.add(href)
        if directory:
            from pathlib import Path
            import hashlib
            from .report_exports import _reference, asset_exclusion
            root = Path(directory).resolve()
            ref = _reference(root/'report.html',root,href)
            path = ref.get('path') if ref and not ref.get('reason') else None
            if path and not asset_exclusion(path,max_bytes=512*1024):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in contents:
                    continue
                contents.add(digest)
        label = labels.get(item.get('label'), item.get('label') or 'Output file')
        if item.get('variant'):
            label += ' · ' + str(item['variant'])
        size = ' · ' + str(item['size_label']) if item.get('size_label') else ''
        priority = '' if item.get('secondary') else ' data-download-priority="primary"'
        (secondary if item.get('secondary') else links).append('<a class="artifact-link"' + priority + ' href="' + html.escape(href, quote=True) + '" title="' + html.escape(href, quote=True)
                                                              + '">' + html.escape(label + size) + '</a>')
    if not links and not secondary:
        return ''
    return ('<section class="reader-section reader-downloads" id="reader-output-files"><h2>Download files</h2>'
            '<p>The CSV opens in spreadsheet software. JSON files contain the full analysis records.</p>'
            '<div class="artifact-links" id="quick-overview-artifacts">' + ''.join(links) + '</div>'
            + ('<details><summary>Additional analysis files</summary><div class="artifact-links">' + ''.join(secondary) + '</div></details>' if secondary else '') + '</section>')


def design_results(payload):
    """Render the saved narrative first, retaining the complete design inventory.

    A design-stage inventory can differ from a separately requested final panel.
    It must not be relabelled as the user's final selection or used to infer a
    new success count. The original scope explanation stays with its table.
    """
    from .report_tables import page_large_tables
    from .report_explorer import explorer_html
    explorer = explorer_html(payload.get('candidate_explorer'))
    saved = payload.get('results_html') or ''
    markdown = (payload.get('overview') or {}).get('report_markdown') or ''
    if not markdown:
        return explorer + (page_large_tables(saved) if saved else '<p>Final results are not recorded.</p>') + design_downloads(payload)
    meta = payload.get('meta') or {}
    from .report_orientation import design_reader_context
    reader_context = payload.get('reader_context')
    collections = (payload.get('outputs') or {}).get('collections') or []
    programs = next((c.get('raw') for c in collections if c.get('key') == 'design_programs'), [])
    saved_program_context = design_reader_context({'design_programs': programs if isinstance(programs, list) else []},
                                                  (payload.get('decisions') or {}).get('strategy_routing'))
    reader_context = {**saved_program_context, **(reader_context if isinstance(reader_context, dict) else {})}
    inventory, separator, _ = saved.partition('<h2>Report and interpretation</h2>')
    body = readable_markdown_html(markdown, payload.get('oligoai_report_dir'), header_request=meta.get('user_request'),
                                  context={**reader_context, 'kind': 'design', 'gene': meta.get('gene_symbol') or meta.get('gene_id'),
                                           'evaluations': payload.get('oligoai_evaluations') or []},
                                  inventory_html=inventory if separator else '', downloads_html=design_downloads(payload), explorer=explorer)
    if saved and not separator:
        # Older or unknown payload layouts are never silently discarded.
        body += '<details><summary>Additional saved results</summary>' + saved + '</details>'
    return page_large_tables(body)


DESIGN_CSS = READER_CSS + """
body{background:#f7f9f7;color:var(--reader-ink)}
.shell{width:100%;max-width:1120px;margin:0 auto;padding:32px 28px 64px;min-width:0}.stack{min-width:0;grid-template-columns:minmax(0,1fr)}
.hero{display:block;background:transparent;color:var(--reader-ink);border:0;border-bottom:1px solid var(--reader-line);border-radius:0;box-shadow:none;padding:0 0 20px;min-height:0}
.hero::before,.hero::after{display:none}.hero h1{font-family:inherit;font-size:2rem;line-height:1.2;margin:0 0 8px;letter-spacing:-.035em}
.hero .eyebrow{color:var(--reader-accent)}.hero-gene-id{color:var(--reader-muted);font-size:.9rem;margin:0 0 8px}
.hero-request{max-width:85ch;color:var(--reader-ink);font-size:.95rem;line-height:1.65;white-space:pre-wrap}
.request-details summary{color:var(--reader-accent);cursor:pointer;font-size:.95rem}
.reader-content details>summary,.request-details>summary,.reader-footer details>summary{list-style:disclosure-closed;display:list-item}
.request-details>summary{list-style-position:inside}
.reader-content details[open]>summary,.request-details[open]>summary,.reader-footer details[open]>summary{list-style-type:disclosure-open}
.reader-content details>summary::-webkit-details-marker,.request-details>summary::-webkit-details-marker,.reader-footer details>summary::-webkit-details-marker{display:inline}
.panel,.panel--strong{background:white;border:1px solid var(--reader-line);border-radius:12px;box-shadow:none}
.tabs{position:static;background:transparent;border:0;box-shadow:none;padding:0;margin:20px 0;gap:8px;flex-wrap:wrap}
.reader-explore{position:relative;font-size:.9rem;max-width:100%}.reader-explore>summary{cursor:pointer;padding:10px 14px;border:1px solid var(--reader-line);border-radius:8px;background:white;list-style:disclosure-closed inside;display:list-item}
.reader-explore[open]>summary{list-style-type:disclosure-open}.reader-detail-tabs{display:flex;flex-wrap:wrap;gap:8px;padding:12px 0 0;max-width:100%}
.tab-btn{background:white;border:1px solid var(--reader-line);border-radius:8px;padding:9px 14px;color:var(--reader-ink);font-family:inherit}
.tab-btn.active{background:var(--reader-accent);color:white;border-color:var(--reader-accent)}
.hero-meta-grid{display:block;min-width:0}.meta-row{display:flex;gap:18px;flex-wrap:wrap;overflow-wrap:anywhere;min-width:0}.meta-row span{min-width:0;max-width:100%;overflow-wrap:anywhere}
.reader-footer{margin-top:24px;font-size:.9rem;color:var(--reader-muted)}.reader-footer summary{cursor:pointer}
.reader-content .artifact-links{display:flex;gap:12px;flex-wrap:wrap}.reader-content .artifact-link{background:#f1f6f2;border:1px solid var(--reader-line);color:var(--reader-accent);font-size:.9rem;max-width:100%;overflow-wrap:anywhere}
#quick-overview-alerts:empty{display:none}#quick-overview-alerts ul{margin-top:8px}
@media(max-width:600px){.shell{padding:20px 12px 40px}.hero h1{font-size:1.6rem}.tabs{gap:6px}.tab-btn{padding:8px 10px;font-size:.85rem}.panel{padding:16px}.meta-row{display:block}}
@media print{.tabs,.reader-footer{display:none}.shell{max-width:none;padding:0}.tab-panel{display:block!important}.is-hidden{display:none!important}}
""" + REPORT_CSS


def composite_document(title, markdown, entries, report_dir):
    """Keep each target's full report linked to its own relative evidence tree."""
    controls = ["<input class='tab-radio' type='radio' name='composite-tabs' id='tab-control-summary' checked>"]
    labels = ["<label class='tab-button' for='tab-control-summary'>Overview</label>"]
    panels = ["<section class='tab-panel reader-content report-content' id='tab-panel-summary'>"
              + readable_markdown_html(markdown, report_dir, context={'kind': 'combined', 'design_labels': [entry['tab_label'] for entry in entries]}) + '</section>']
    rules = ["#tab-control-summary:checked ~ #tab-panel-summary{display:block}",
             "#tab-control-summary:checked ~ .tab-list label[for='tab-control-summary']{background:var(--reader-accent);color:white}",
             "#tab-control-summary:focus-visible ~ .tab-list label[for='tab-control-summary']{outline:3px solid var(--reader-accent);outline-offset:3px}"]
    counts = {}
    for index, entry in enumerate(entries, 1):
        label = entry['tab_label']; counts[label] = counts.get(label, 0) + 1
        if sum(other['tab_label'] == label for other in entries) > 1:
            label += f" · {counts[label]}"
        control, panel = f'tab-control-design-{index}', f'tab-panel-design-{index}'
        controls.append(f"<input class='tab-radio' type='radio' name='composite-tabs' id='{control}'>")
        labels.append(f"<label class='tab-button' for='{control}'>" + html.escape(label) + '</label>')
        rules.extend([f'#{control}:checked ~ #{panel}{{display:block}}',
                      f"#{control}:checked ~ .tab-list label[for='{control}']{{background:var(--reader-accent);color:white}}",
                      f"#{control}:focus-visible ~ .tab-list label[for='{control}']{{outline:3px solid var(--reader-accent);outline-offset:3px}}"])
        href = html.escape(entry['html_href'], quote=True)
        links = f"<a href='{href}' target='_blank' rel='noopener'>Open full report</a>"
        if entry.get('md_href'):
            links += "<a href='" + html.escape(entry['md_href'], quote=True) + "'>Download text report</a>"
        panels.append(f"<section class='tab-panel reader-content' id='{panel}'>"
                      + '<h2>' + html.escape(label) + '</h2>'
                      + '<details><summary>Design request</summary><p>' + html.escape(entry['title']) + '</p></details>'
                      + '<nav class="reader-nav">' + links + '</nav>'
                      + f"<iframe class='child-report-frame' loading='lazy' title='{html.escape(label, quote=True)}' src='{href}'></iframe></section>")
    style = (".tab-radio{position:absolute;width:1px;height:1px;opacity:0}.tab-list{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}"
             ".tab-button{padding:8px 15px;border:1px solid #dbe4de;border-radius:8px;cursor:pointer;background:white}.tab-panel{display:none}"
             ".child-report-frame{width:100%;height:80vh;border:1px solid #dbe4de;border-radius:8px}"
             + ''.join(rules))
    display_title = concise_title(title, fallback='Combined ASO report')
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            + '<title>' + html.escape(display_title) + '</title><style>' + READER_CSS + style + '</style></head><body>'
            + '<a class="reader-skip" href="#reader-main">Skip to results</a><div class="reader-page"><header class="reader-header"><p class="reader-brand">MODAL-ASO</p><h1>'
            + html.escape(display_title) + '</h1><p class="reader-subtitle">Read the combined results below, or open a report for an individual target.</p>'
            + ('<details><summary>Your request</summary><p>' + html.escape(str(title)) + '</p></details>' if display_title != title else '')
            + '</header><main id="reader-main" tabindex="-1">' + ''.join(controls) + '<nav class="tab-list" aria-label="Reports by target">' + ''.join(labels)
            + '</nav>' + ''.join(panels) + '</main></div>' + READER_SCRIPT + '</body></html>')
