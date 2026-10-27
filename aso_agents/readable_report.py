"""Reader-facing HTML over saved reports; never rewrite the scientific source.

Short explanations introduce verified saved counts and terms. Free scientific
prose, table records, identifiers, units and sequences remain available intact.
Unknown sections stay expanded.
This module deliberately has no model, prediction or selection dependencies.
"""
from __future__ import annotations

import html
import re

from .report_presentation import TABLE_CSS, markdown_html


READER_CSS = TABLE_CSS + """
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
@media(max-width:600px){.reader-facts{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.reader-fact{padding:12px}.reader-fact-value{font-size:1.6rem}.reader-fact-label{font-size:.9rem}.reader-orientation h2{font-size:1.35rem}}
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
    "conditional exon-skipping sequence consequences": "Predicted effects of exon skipping",
    "practical interpretation": "What the results mean",
    "outcome summary": "Results at a glance",
    "candidate sequence audit and reconciliation": "Candidate sequence checks",
    "full observation membership": "All comparison results",
    "complete audit table": "All sequence checks",
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
}

# Fold only explicitly named methods/administration, never arbitrary prose or
# an unfamiliar section that might contain an important result or caveat.
DETAIL_HEADINGS = {
    "method", "methods", "general result", "main files to open", "task outcomes",
    "how to read the rankings", "where users should look to choose asos",
    "references and search scope", "off-target search references",
    "selected-panel off-target search budget", "selection and evidence",
}
RESULT_HEADINGS = {
    "outcome summary", "practical interpretation", "bottom line", "selected physical panel",
    "design run status", "recorded panel comparison", "lead records (recorded rank 1)",
}
LIMIT_HEADINGS = {"caveats", "limitations", "main uncertainty", "what remains uncertain"}

TABLE_LABELS = {
    'id': 'ID', 'set': 'Set', 'sequence': 'Sequence',
    'target_score': 'Target score', 'offtarget_hits': 'Off-target matches',
    'source_row': 'Input row', 'source_row_number': 'Input row',
    'pooled_nondominated_frontier': 'Nondominated across all sets',
    'dominated_by_ids': 'Candidates with better compared measures',
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
}
# Redundant arithmetic and input-row bookkeeping can be opened on demand.
# Unknown columns and all decision/result/limitation columns remain visible.
DETAIL_COLUMNS = {
    'source_row', 'source_row_number', 'claimed_length', 'claimed_gc', 'gc_count',
    'recomputed_gc_fraction_exact', 'recomputed_gc_fraction_3dp',
    'longest_homopolymer_base', 'claimed_gc_rounded_3dp', 'exclusion_reason_count',
    'set_total_count', 'set_frontier_count', 'set_dominated_count', 'set_unknown_count',
}

STATUS_LABELS = {
    'completed_with_candidates': 'Candidates produced',
    'completed_no_candidates': 'No candidates produced',
    'outputs_present': 'Output files available',
    'not_predicted': 'Prediction unavailable',
    'not_applicable': 'Not applicable',
    'inference_error': 'Prediction failed',
}


def _reason_label(value):
    """Decode explicit conditions, retaining arbitrary thresholds exactly."""
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
                if key == 'native rank / scope' and (rank := re.fullmatch(r'(\d+|unavailable|not recorded)\s+/\s+(.+)', raw)):
                    rank_label = rank[1] if rank[1].isdigit() else 'No rank supplied'
                    return ('<td data-recorded-value="' + html.escape(raw, quote=True)
                            + '"><details class="reader-rank"><summary>' + rank_label
                            + ' · group details</summary><p>Design group: <code>'
                            + html.escape(rank[2]) + '</code></p></details></td>')
                if key in {'status', 'outcome', 'retention_status'}:
                    label = STATUS_LABELS.get(raw, raw)
                elif key in {'exclusion reason', 'exclusion_reasons'}:
                    label = _reason_label(raw)
                else:
                    label = raw
                if label == raw:
                    return cell_match[0]
                # The source code remains machine-checkable and available in the
                # tooltip and unchanged text report. Numeric cells never change.
                return '<td data-recorded-value="' + html.escape(raw, quote=True) + '"><abbr title="Recorded value: ' + html.escape(raw, quote=True) + '">' + html.escape(label) + '</abbr></td>'
            return re.sub(r'<td>(.*?)</td>', cell, row_match[0], flags=re.S)
        return re.sub(r'<tr>.*?</tr>', row, table, flags=re.S)
    fragment = re.sub(r'<table>.*?</table>', values, fragment, flags=re.S)
    def header(match):
        value = html.unescape(match[1]); key = value.strip('`').casefold()
        if key not in TABLE_LABELS:
            return match[0]
        css = ' class="reader-detail-column"' if key in DETAIL_COLUMNS else ''
        return '<th' + css + ' title="' + html.escape(value, quote=True) + '">' + html.escape(TABLE_LABELS[key]) + '</th>'
    return re.sub(r'<th>([^<]*)</th>', header, fragment)

TERMS = (
    (r"\bASOs?\b|antisense", "ASO", "A short synthetic strand designed to bind RNA. A candidate ASO is a proposed option for further study."),
    (r"\bRNA\b|transcript", "RNA / transcript", "RNA carries or helps use genetic instructions. A transcript is one RNA product of a gene; different versions can contain different regions."),
    (r"splic", "Splicing", "How pieces of RNA are joined. Changes in splicing can change which RNA version is produced."),
    (r"predict|estimate", "Prediction", "A result calculated by a model. It is different from an effect measured in an experiment."),
    (r"\bisoform\w*\b", "Isoform", "A version of an RNA transcript. Coverage of one version does not necessarily mean coverage of all versions."),
    (r"off[-_ ]target", "Off-target match", "A sequence match outside the intended target. The reference and mismatch settings limit which matches were checked; a match alone does not establish a biological effect."),
    (r"\bGC(?:_|\b)", "GC fraction", "The fraction of a sequence made up of G and C bases. A fraction of 0.30 is 30%."),
    (r"homopolymer", "Homopolymer", "A run of the same base, such as AAAA."),
    (r"nondominated|pareto", "Nondominated candidate", "No other candidate in the stated comparison is at least as good on every compared measure and better on at least one. This is not a claim of overall biological superiority."),
    (r"\brank\b|selection_rank", "Rank", "The order recorded for a particular candidate set. Ranks from different sets or scoring methods may not be comparable."),
    (r"\b(?:unavailable|not_predicted|not recorded)\b", "Unavailable result", "A missing measurement or prediction. It is not a zero, a negative finding, or evidence that the ASO does not work."),
    (r"\bN/?A\b", "NA / N/A", "The report supplies no value in that field. It may be unavailable or not applicable; check the accompanying notes. It does not mean zero."),
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


def readable_markdown_html(markdown, report_dir=None, *, include_tools=True, header_title=None, header_request=None, context=None):
    """Reorganize saved text for reading without editing its evidence blocks.

    Reordering is limited to known result/limitation sections. Every original
    body is rendered once. Unknown headings are visible in their original order.
    """
    sections = _sections(markdown)
    visible, details, intro, downloads = [], [], [], []
    for index, section in enumerate(sections):
        heading = section['heading']; key = heading.casefold()
        label = HEADING_LABELS.get(key, heading)
        body = friendly_tables(markdown_html('\n'.join(section['lines']), report_dir))
        section_id = f'reader-section-{index}'
        if not heading or section['level'] == 1:
            content = friendly_tables(_intro_html(section['lines'], report_dir, details, header_request))
            if heading and label != header_title and key not in {'run output summary', 'sandbox summary'}:
                content = '<h2>' + html.escape(label) + '</h2>' + content
            intro.append((index, section_id, label or 'Overview', content))
        elif key in DETAIL_HEADINGS:
            details.append('<details id="' + section_id + '"><summary>' + html.escape(label)
                           + '</summary><div>' + body + '</div></details>')
        else:
            cls = ' reader-limits' if key in LIMIT_HEADINGS else ' reader-downloads' if key == 'requested output files' else ''
            content = '<h2>' + html.escape(label) + '</h2>' + body
            item = (index, section_id, label, '<section class="reader-section' + cls + '" id="' + section_id + '">' + content + '</section>')
            if key == 'requested output files':
                downloads.append(item)
            else:
                visible.append((0 if key in RESULT_HEADINGS else 1 if key in LIMIT_HEADINGS else 2, item))
    visible.sort(key=lambda entry: (entry[0], 0 if entry[1][2] == 'Main finding' else 1, entry[1][0]))
    blocks = [item for _, item in visible]
    nav = []
    if blocks:
        nav.append(('<a href="#' + blocks[0][1] + '">Results</a>'))
    limits = next((item for priority, item in visible if priority == 1), None)
    if limits:
        nav.append('<a href="#' + limits[1] + '">Limitations</a>')
    if downloads:
        nav.append('<a href="#' + downloads[0][1] + '">Downloads</a>')
    if details:
        nav.append('<a href="#reader-details">Methods and details</a>')
    terms = glossary_html(str(markdown or ''))
    if terms:
        nav.append('<a href="#reader-terms">Explain report terms</a>')
    tools = ('<div class="reader-tools"><h2 id="reader-findings">Findings and data</h2>'
             '<button type="button" class="reader-expand">Expand all details</button></div>') if include_tools else '<span id="reader-findings"></span>'
    # A free-form introduction may itself be the only scientific summary.
    lead = ''.join('<section class="reader-section" id="' + item[1] + '">' + item[3] + '</section>' for item in intro)
    detail_html = ('<section class="reader-section" id="reader-details"><h2>Methods and details</h2>' + ''.join(details) + '</section>') if details else ''
    from .report_orientation import build_orientation, orientation_html
    orientation = orientation_html(build_orientation(sections, context), findings='reader-findings',
                                   downloads=downloads[0][1] if downloads else None,
                                   limitations=limits[1] if limits else None)
    return (orientation + tools + ('<nav class="reader-nav" aria-label="In this report">' + ''.join(nav) + '</nav>' if nav else '')
            + lead + ''.join(item[3] for item in blocks) + ''.join(item[3] for item in downloads)
            + detail_html + terms)


READER_SCRIPT = """<script>
// In-page links can point to a folded explanation or a nested source section.
function revealReaderAnchor(){
  let anchor=location.hash.slice(1);try{anchor=decodeURIComponent(anchor);}catch(_error){}
  const target=document.getElementById(anchor);
  if(!target)return;
  for(let node=target;node;node=node.parentElement){if(node.tagName==='DETAILS')node.open=true;}
}
addEventListener('hashchange',revealReaderAnchor);revealReaderAnchor();
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
</script>"""


def concise_title(title, *, fallback='ASO analysis'):
    value = str(title or '').strip()
    if not value or len(value) > 100 or '\n' in value:
        return fallback
    return HEADING_LABELS.get(value.casefold(), value)


def basic_document(title, markdown, report_dir=None):
    display_title = concise_title(title)
    if display_title == 'ASO analysis':
        sections = _sections(markdown)
        first = next((section['heading'] for section in sections if section['level'] == 1), '')
        if first.casefold() in HEADING_LABELS and first.casefold() not in {'run output summary', 'sandbox summary'}:
            display_title = HEADING_LABELS[first.casefold()]
    body = readable_markdown_html(markdown, report_dir, header_title=display_title)
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            '<title>' + html.escape(display_title) + '</title><style>' + READER_CSS + '</style></head><body>'
            '<div class="reader-page"><header class="reader-header"><p class="reader-brand">MODAL-ASO</p><h1>'
            + html.escape(display_title) + '</h1><p class="reader-subtitle">Results, interpretation and supporting information.</p>'
            + ('<details><summary>Full report title</summary><p>' + html.escape(str(title)) + '</p></details>' if display_title != title and len(str(title or '')) > 100 else '')
            + '</header><main class="reader-content report-content">' + body + '</main></div>' + READER_SCRIPT + '</body></html>')


def design_results(payload):
    """Render the saved narrative first, retaining the complete design inventory.

    A design-stage inventory can differ from a separately requested final panel.
    It must not be relabelled as the user's final selection or used to infer a
    new success count. The original scope explanation stays with its table.
    """
    saved = payload.get('results_html') or ''
    markdown = (payload.get('overview') or {}).get('report_markdown') or ''
    if not markdown:
        return saved or '<p>Final results are not recorded.</p>'
    meta = payload.get('meta') or {}
    body = readable_markdown_html(markdown, payload.get('oligoai_report_dir'), header_request=meta.get('user_request'),
                                  context={'kind': 'design', 'gene': meta.get('gene_symbol') or meta.get('gene_id'),
                                           'evaluations': payload.get('oligoai_evaluations') or []})
    inventory, separator, _ = saved.partition('<h2>Report and interpretation</h2>')
    if inventory and separator:
        body += ('<details><summary>ASOs produced during design</summary>' + inventory + '</details>')
    elif saved:
        # Older or unknown payload layouts are never silently discarded.
        body += '<details><summary>Additional saved results</summary>' + saved + '</details>'
    return body


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
"""


def composite_document(title, markdown, entries, report_dir):
    """Keep each target's full report linked to its own relative evidence tree."""
    controls = ["<input class='tab-radio' type='radio' name='composite-tabs' id='tab-control-summary' checked>"]
    labels = ["<label class='tab-button' for='tab-control-summary'>Overview</label>"]
    panels = ["<section class='tab-panel reader-content report-content' id='tab-panel-summary'>"
              + readable_markdown_html(markdown, report_dir, context={'kind': 'combined', 'design_labels': [entry['tab_label'] for entry in entries]}) + '</section>']
    rules = ["#tab-control-summary:checked ~ #tab-panel-summary{display:block}",
             "#tab-control-summary:checked ~ .tab-list label[for='tab-control-summary']{background:#23624e;color:white}",
             "#tab-control-summary:focus-visible ~ .tab-list label[for='tab-control-summary']{outline:3px solid #23624e;outline-offset:3px}"]
    counts = {}
    for index, entry in enumerate(entries, 1):
        label = entry['tab_label']; counts[label] = counts.get(label, 0) + 1
        if sum(other['tab_label'] == label for other in entries) > 1:
            label += f" · {counts[label]}"
        control, panel = f'tab-control-design-{index}', f'tab-panel-design-{index}'
        controls.append(f"<input class='tab-radio' type='radio' name='composite-tabs' id='{control}'>")
        labels.append(f"<label class='tab-button' for='{control}'>" + html.escape(label) + '</label>')
        rules.extend([f'#{control}:checked ~ #{panel}{{display:block}}',
                      f"#{control}:checked ~ .tab-list label[for='{control}']{{background:#23624e;color:white}}",
                      f"#{control}:focus-visible ~ .tab-list label[for='{control}']{{outline:3px solid #23624e;outline-offset:3px}}"])
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
            + '<div class="reader-page"><header class="reader-header"><p class="reader-brand">MODAL-ASO</p><h1>'
            + html.escape(display_title) + '</h1><p class="reader-subtitle">Read the combined results below, or open a report for an individual target.</p>'
            + ('<details><summary>Your request</summary><p>' + html.escape(str(title)) + '</p></details>' if display_title != title else '')
            + '</header><main>' + ''.join(controls) + '<nav class="tab-list" aria-label="Reports by target">' + ''.join(labels)
            + '</nav>' + ''.join(panels) + '</main></div>' + READER_SCRIPT + '</body></html>')
