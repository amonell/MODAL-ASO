"""Human-readable views of structured evidence, without changing saved records.

Only complete JSON/Python collections are decoded. Scientific strings and
identifiers stay literal; BED coordinates require an explicitly named convention.
The browser renderer uses the same labels as the static report.
"""
from __future__ import annotations

import ast
from decimal import Decimal
import html
import json
import math
import re


FIELD_LABELS = {
    'chrom': 'Reference sequence', 'chr': 'Chromosome',
    'coordinate_system': 'Position numbering', 'coordinate_space': 'Reference type',
    'original_row_bed0_half_open': 'Original position (zero-based; end excluded)',
    'mapped_row_bed0_half_open': 'Mapped position (zero-based; end excluded)',
    'binding_intervals_bed0_half_open': 'Binding positions (zero-based; end excluded)',
    'physical_binding_intervals_bed0': 'Binding positions (zero-based; end excluded)',
    'intervals_bed0_half_open': 'Positions (zero-based; end excluded)',
    'reference_spans_bed0_half_open': 'Reference spans (zero-based; end excluded)',
    'source_intervals_bed0_half_open': 'Original reference positions (zero-based; end excluded)',
    'aso_overlap_requirement': 'Required overlap',
    'gc_fraction': 'GC fraction', 'transcript_id': 'Transcript ID',
    'gene_id': 'Gene ID', 'gene_symbol': 'Gene', 'target_build': 'Reference assembly',
    'original_build': 'Original reference assembly', '__truncated__': 'Additional records',
    'physical binding interval(s), bed0': 'Binding positions (zero-based; end excluded)',
    'binding interval(s), bed0': 'Binding positions (zero-based; end excluded)',
    'searched_gene_span_bed0': 'Searched gene region (zero-based; end excluded)',
    'source_rows_bed0_half_open': 'Dataset regions (zero-based; end excluded)',
    'reference_feature_rows_bed0_half_open': 'Requested RNA regions (zero-based; end excluded)',
    'projected_rows_bed0_half_open': 'Overlapping regions (zero-based; end excluded)',
    'feature_scope': 'Requested RNA region',
    'off_target_positions_truncated': 'Location list shortened for display',
    'off_target_position_preview_limit': 'Locations shown per mismatch group',
    'off_target_search_method': 'Search method',
    'off_target_gene_annotation_status': 'Gene annotation coverage',
    'off_target_expression_weighted_risk': 'Weighted expression of matching RNAs',
    'expression_weighted_risk': 'Weighted expression of matching RNAs',
    'off_target_expression_sum': 'Expression of matching RNAs',
    'off_target_expression_sum_2mm': 'Expression of RNAs matched with two mismatches',
}

# Explicit vocabulary, rather than replacing underscores in arbitrary values.
VALUE_LABELS = {
    'monitoring_only': 'Changes recorded for context', 'observe': 'Record changes for context',
    'vendor_notation_unsupported': 'Supplier ordering notation unavailable; use the full chemistry specification',
    'MORPHOLINO': 'Morpholino', 'PHOSPHORODIAMIDATE': 'Phosphorodiamidate',
    'uncapped_overflow_recovery': 'Complete search of repetitive sequences',
    'full_containment': 'Entire ASO binding site inside the region',
    'any_overlap': 'ASO binding site overlaps the region',
    'candidates_available': 'Candidates available',
    'genomic_pre_mrna': 'Pre-mRNA (genomic reference)',
    'mature_mrna': 'Mature mRNA', 'mature_rna': 'Mature RNA',
    '0based_half_open': 'Zero-based; end excluded',
    '1based_inclusive': 'One-based; both ends included',
    'user_provided_intervals': 'Regions supplied in the request',
    'native_reference': 'Already on the requested reference assembly',
    'materialized': 'Reference regions prepared',
    'transcript_oriented_target_binding_site': 'Target RNA binding site',
    'top_ranked_after_required_evidence_and_constraints_per_site':
        'Highest-ranked ASOs at each site after the required checks and filters',
    'selection_rank': 'Selection rank',
    'Off_target_exact_count': 'Exact off-target match count',
    'repeat_hit_count_exact': 'Alternate exact repeat locations',
    'off_target_gene_count': 'Number of other genes matched',
    'Times_mapped': 'Number of recorded mappings',
    'max_binding_sites': 'Maximum allowed exact matches',
    'complete_within_declared_scope': 'Search completed within the recorded limits',
    'incomplete_or_unavailable': 'Incomplete or missing',
    'not_requested': 'Outside this analysis', 'not_applicable': 'Not applicable',
    'not_predicted': 'Prediction unavailable', 'inference_error': 'Prediction failed',
    'outputs_present': 'Output files available',
    'completed_with_candidates': 'Candidates produced',
    'completed_no_candidates': 'No candidates produced',
    'splice_modulation': 'Change RNA splicing',
    'translation_blocking': 'Block translation',
    'direction_supported': 'Prediction supports the requested direction',
    'partial_direction_support': 'Partial prediction support for the requested direction',
    '5prime_to_3prime': '5′→3′', 'bed0_half_open': 'Zero-based; end excluded',
    'gapmer_tiling': 'Gapmer candidate generation', 'steric_tiling': 'Steric blocker candidate generation',
    'projected_from_transcript_exons': 'Mapped from transcript exons',
    'native_genomic_reference': 'Genomic reference', 'not_evaluated': 'Not evaluated',
    'mapping_inventory_unavailable': 'Match-location records unavailable',
    'available_gene_transcript_catalog': 'Available transcripts for this gene',
    'recorded_pair_predictions': 'Recorded paired predictions',
    'explicit_exon_objective': 'Requested exon outcome',
    'selected_design_transcripts': 'Transcripts used for design',
    'reference_scope': 'Reference scope', 'mixed_or_partial_predictions': 'Mixed or incomplete predictions',
    'splice_mod': 'Splicing change',
    'scored_splice_response_site': 'Splice site evaluated by the model',
    'rbp_binding': 'RNA-binding protein sites',
    'mirna_binding': 'miRNA-binding sites',
    'sites_found': 'Binding sites found',
    'intersects_feature': 'Overlaps the requested RNA region',
    'outside_feature_or_opposite_strand': 'Outside the requested RNA region or on the opposite strand',
}

FILE_LABELS = {
    'design_panel.csv': 'Selected ASOs (CSV)', 'design_panel.md': 'Selected ASOs and binding evidence',
    'generated_aso_set.json': 'Complete ASO records (JSON)',
    'generated_filter_trace.json': 'Candidate filtering history (JSON)',
    'requested_binding_site_sets.json': 'Candidates in the requested regions (JSON)',
    'region_constraint_evidence.json': 'Requested region evidence (JSON)',
    'reference_transcript_catalog.json': 'Reference transcripts (JSON)',
    'strategy_router_summary.json': 'Splicing strategy rankings (JSON)',
    'scored_tiles.xlsx': 'All scored candidates (Excel)',
    'scored_tiles.json': 'All scored candidates (JSON)',
    'tiles.json': 'Initial candidate records (JSON)',
}

IDENTITY_KEYS = {'id', 'candidate', 'sequence', 'aso', 'name', 'label', 'chemistry', 'motif'}
VALUE_CSS = """
.reader-value-record{margin:0;min-width:0}
.reader-value-record>div{display:grid;grid-template-columns:minmax(130px,1fr) minmax(0,2fr);gap:6px 18px;padding:8px 0;border-bottom:1px solid #dbe4de}
.reader-value-record dt{font-weight:600;overflow-wrap:anywhere}.reader-value-record dd{margin:0;min-width:0;overflow-wrap:anywhere}
.reader-value-record .reader-value-record>div{grid-template-columns:minmax(0,1fr);gap:2px}
.reader-value-list{margin:0;padding-left:22px}.reader-value-list>li+li{margin-top:8px}
.reader-value-details{margin:6px 0!important;padding:8px 12px!important;min-width:0}
.reader-value-details>summary{cursor:pointer;font-weight:500}
.reader-value-inline{overflow-wrap:anywhere}
.reader-value-block{margin:12px 0;overflow-wrap:anywhere}
@media(max-width:600px){.reader-value-record>div{grid-template-columns:1fr;gap:2px}.reader-value-record dd{padding-left:10px}}
"""


def field_label(key):
    text = str(key)
    if text in FIELD_LABELS or text.lower() in FIELD_LABELS:
        return FIELD_LABELS.get(text, FIELD_LABELS.get(text.lower()))
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text.replace('_', ' ')).strip()
    return text[:1].upper() + text[1:]


def identity_key(key):
    key = str(key).lower().replace(' ', '_')
    return key in IDENTITY_KEYS or key.endswith(('_id', '_ids', '_sequence', '_sequences', '_notation'))


def parse_collection(text):
    """Return a collection or None; never evaluate code or guess malformed data."""
    if not isinstance(text, str) or not text.strip().startswith(('[', '{')):
        return None
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError, RecursionError):
            return None
    return value if isinstance(value, (dict, list)) else None


def bed_positions(value, key):
    """Keep BED0 endpoints exactly as recorded, including BED6 metadata."""
    if 'bed0' not in str(key).lower():
        return None
    rows = value if isinstance(value, list) and value and isinstance(value[0], list) else [value]
    labels = []
    for row in rows:
        if (not isinstance(row, list) or len(row) not in (3, 6)
                or not isinstance(row[0], str) or type(row[1]) is not int or type(row[2]) is not int
                or not 0 <= row[1] < row[2]):
            return None
        label = f'{row[0]}:{row[1]}–{row[2]}'
        if len(row) == 6:
            if not isinstance(row[3], str) or not isinstance(row[4], (int, float)) or row[5] not in ('+', '-', '.'):
                return None
            label += f'; name: {row[3]}; score: {row[4]}; strand: {row[5]}'
        labels.append(label)
    return '; '.join(labels) + ' (zero-based; end excluded)' if labels else None


def scalar_label(value, key=''):
    if value is None:
        return 'Not reported'
    if value is True:
        return 'Yes'
    if value is False:
        return 'No'
    if isinstance(value, float) and math.isfinite(value):
        return format(Decimal(str(value)).normalize(), 'f')
    text = str(value)
    return text if identity_key(key) else VALUE_LABELS.get(text, text)


def value_text(value, key=''):
    value = presentation_value(value, key)
    if (position := bed_positions(value, key)) is not None:
        return position
    if isinstance(value, dict):
        return '; '.join(f'{field_label(k)}: {value_text(v, k)}' for k, v in value.items()) or 'No details recorded'
    if isinstance(value, list):
        return '; '.join(value_text(v, key) for v in value) or 'None listed'
    return scalar_label(value, key)


def value_html(value, key='', *, inline=False):
    value = presentation_value(value, key)
    if (position := bed_positions(value, key)) is not None:
        return html.escape(position)
    if isinstance(value, dict):
        if not value:
            return 'No details recorded'
        if inline:
            return '<span class="reader-value-inline">' + '; '.join(
                '<span><strong>' + html.escape(field_label(k)) + ':</strong> ' + value_html(v, k, inline=True) + '</span>'
                for k, v in value.items()) + '</span>'
        entries = []
        for k, v in value.items():
            v = presentation_value(v, k)
            body = value_html(v, k)
            if isinstance(v, (dict, list)) and v and bed_positions(v, k) is None:
                count = len(v)
                noun = 'field' if isinstance(v, dict) else 'item'
                body = '<details class="reader-value-details"><summary>' + str(count) + ' ' + noun + (
                    's' if count != 1 else '') + '</summary>' + body + '</details>'
            entries.append('<div><dt>' + html.escape(field_label(k)) + '</dt><dd>' + body + '</dd></div>')
        return '<dl class="reader-value-record">' + ''.join(entries) + '</dl>'
    if isinstance(value, list):
        if not value:
            return 'None listed'
        if inline:
            return '<span class="reader-value-inline">' + '; '.join('<span>' + value_html(v, key, inline=True) + '</span>' for v in value) + '</span>'
        return '<ul class="reader-value-list">' + ''.join('<li>' + value_html(v, key) + '</li>' for v in value) + '</ul>'
    return html.escape(scalar_label(value, key))


def presentation_value(value, key=''):
    if isinstance(value, str) and (not identity_key(key) or str(key).lower().endswith('_ids')):
        parsed = parse_collection(value)
        if parsed is not None:
            return parsed
    return value


def recorded_html(raw, value, key='', *, inline=False):
    tag = 'span' if inline else 'div'
    return ('<' + tag + ' class="reader-value-' + ('inline' if inline else 'block')
            + '" data-recorded-value="' + html.escape(raw, quote=True) + '">'
            + value_html(value, key, inline=inline) + '</' + tag + '>')


def readable_prose(fragment):
    """Format complete evidence records in generated HTML, leaving code literal.

    Tables are handled by their own field-aware formatter. Links, chemical
    strings, code examples, identifiers and malformed collections are retained.
    """
    def paragraph(match):
        raw = match[1].removeprefix('<code>').removesuffix('</code>')
        if '<' in raw:
            return match[0]
        raw = html.unescape(raw)
        value = parse_collection(raw)
        return recorded_html(raw, value) if value is not None else match[0]
    token = re.compile(r'(<table\b.*?</table>|<pre><code>.*?</code></pre>|<p>.*?</p>|<code>.*?</code>|<[^>]+>)', re.S)
    enum_pattern = re.compile(r'(?<![\w./-])(' + '|'.join(map(re.escape, sorted(VALUE_LABELS, key=len, reverse=True))) + r')(?![\w/-]|\.[\w])')

    def prose(text):
        text = html.unescape(text)
        output, pos = [], 0
        decoder = json.JSONDecoder()
        for match in re.finditer(r'[\[{]', text):
            if match.start() < pos:
                continue
            try:
                value, end = decoder.raw_decode(text[match.start():])
            except (ValueError, RecursionError):
                continue
            if not isinstance(value, (dict, list)):
                continue
            prefix = text[pos:match.start()]
            output.append(words(prefix))
            key = 'bed0' if re.search(r'\bBED0\b|0-based half-open', text[:match.start()], re.I) else ''
            raw = text[match.start():match.start() + end]
            output.append(recorded_html(raw, value, key, inline=True))
            pos = match.start() + end
        output.append(words(text[pos:]))
        return ''.join(output)

    def words(text):
        # A local filesystem prefix is useful to software, but the file name
        # identifies the evidence for a reader. Do not treat vendor /5.../
        # chemistry strings as paths. Exact paths remain in the saved record.
        path_pattern = r'(?<![\w:/])((?:/(?:mnt|home|tmp|srv|data|output)/|artifacts/)[^\s<>]+)'
        pieces, position = [], 0
        for match in re.finditer(path_pattern, text):
            pieces.append(enums(text[position:match.start()]))
            raw = match[0].rstrip('.,;:')
            label = FILE_LABELS.get(raw.rsplit('/', 1)[-1], raw.rsplit('/', 1)[-1] or raw)
            pieces.append('<span data-recorded-value="' + html.escape(raw, quote=True) + '">' + html.escape(label) + '</span>' + html.escape(match[0][len(raw):]))
            position = match.end()
        pieces.append(enums(text[position:]))
        return ''.join(pieces)

    def enums(text):
        return enum_pattern.sub(lambda m: '<span data-recorded-value="' + html.escape(m[0], quote=True) + '">'
                                + html.escape(VALUE_LABELS[m[0]]) + '</span>', html.escape(text))

    parts = token.split(fragment)
    for index, part in enumerate(parts):
        if not part:
            continue
        if part.startswith('<p>'):
            transformed = paragraph(re.fullmatch(r'<p>(.*?)</p>', part, re.S))
            parts[index] = transformed if transformed != part else '<p>' + readable_prose(part[3:-4]) + '</p>'
        elif part.startswith(('<pre><code>', '<code>')):
            fenced = part.startswith('<pre>')
            raw = html.unescape(re.sub(r'^<pre><code>|</code></pre>$|^<code>|</code>$', '', part))
            value = parse_collection(raw)
            if value is not None:
                parts[index] = recorded_html(raw, value, inline=not fenced)
            elif not fenced and raw in VALUE_LABELS:
                parts[index] = '<span data-recorded-value="' + html.escape(raw, quote=True) + '">' + html.escape(VALUE_LABELS[raw]) + '</span>'
            elif not fenced and (raw.rsplit('/', 1)[-1] in FILE_LABELS or re.match(r'^(?:/(?:mnt|home|tmp|srv|data|output)/|artifacts/)[^\s]+$', raw)):
                label = FILE_LABELS.get(raw.rsplit('/', 1)[-1], raw.rsplit('/', 1)[-1] or raw)
                parts[index] = '<span data-recorded-value="' + html.escape(raw, quote=True) + '">' + html.escape(label) + '</span>'
        elif not part.startswith('<'):
            parts[index] = prose(part)
    return ''.join(parts)


def browser_value_renderer(payload=None):
    """Self-contained renderer for the report's dynamically built detail views."""
    # Older CSV-derived fields contain Python list literals. Decode them safely
    # on the server; the browser never evaluates a string as code. The original
    # report-data payload and its strings remain unchanged.
    legacy = {}
    pending = [payload]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str) and item.strip().startswith(('[', '{')):
            try:
                parsed_json = json.loads(item)
                if isinstance(parsed_json, (dict, list)):
                    pending.append(parsed_json)
            except (ValueError, RecursionError):
                parsed = parse_collection(item)
                if parsed is not None:
                    try:
                        json.dumps(parsed)
                    except (TypeError, ValueError):
                        continue
                    legacy[item] = parsed
    config = json.dumps({'fields': FIELD_LABELS, 'values': VALUE_LABELS, 'identities': sorted(IDENTITY_KEYS),
                         'legacyCollections': legacy}, ensure_ascii=True).replace('</', '<\\/')
    return 'const reportValueLabels = ' + config + ';\n' + VALUE_SCRIPT


VALUE_SCRIPT = r'''
function reportOwnLabel(labels, key) {
  return Object.prototype.hasOwnProperty.call(labels, key) ? labels[key] : undefined;
}
function reportFieldLabel(key) {
  const text = String(key);
  return reportOwnLabel(reportValueLabels.fields, text) || reportOwnLabel(reportValueLabels.fields, text.toLowerCase()) ||
    text.replace(/_/g, ' ').replace(/([a-z0-9])([A-Z])/g, '$1 $2').trim().replace(/^./, c => c.toUpperCase());
}
function reportIdentityKey(key) {
  key = String(key).toLowerCase().replace(/ /g, '_');
  return reportValueLabels.identities.includes(key) || /(_id|_ids|_sequence|_sequences|_notation)$/.test(key);
}
function reportScalarLabel(value, key = '') {
  if (value === null || value === undefined) return 'Not reported';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (Object.is(value, -0)) return '-0';
    const text = String(value);
    if (!text.includes('e')) return text;
    const [mantissa, exponent] = text.split('e');
    const sign = mantissa.startsWith('-') ? '-' : '';
    const unsigned = mantissa.replace('-', '');
    const parts = unsigned.split('.');
    const digits = parts.join('');
    const point = parts[0].length + Number(exponent);
    return sign + (point <= 0 ? '0.' + '0'.repeat(-point) + digits :
      point >= digits.length ? digits + '0'.repeat(point - digits.length) : digits.slice(0, point) + '.' + digits.slice(point));
  }
  const text = String(value);
  return reportIdentityKey(key) ? text : (reportOwnLabel(reportValueLabels.values, text) || text);
}
function reportParseValue(value, key = '') {
  if (reportIdentityKey(key) && !String(key).toLowerCase().endsWith('_ids')) return value;
  if (typeof value === 'string' && /^[\s]*[\[{]/.test(value)) {
    try { const parsed = JSON.parse(value); if (parsed && typeof parsed === 'object') return parsed; } catch (_) {}
    if (Object.prototype.hasOwnProperty.call(reportValueLabels.legacyCollections, value)) return reportValueLabels.legacyCollections[value];
  }
  return value;
}
function reportBedPositions(value, key) {
  if (!String(key).toLowerCase().includes('bed0')) return null;
  const rows = Array.isArray(value) && Array.isArray(value[0]) ? value : [value];
  const labels = [];
  for (const row of rows) {
    if (!Array.isArray(row) || ![3, 6].includes(row.length) || typeof row[0] !== 'string' ||
        !Number.isSafeInteger(row[1]) || !Number.isSafeInteger(row[2]) || !(0 <= row[1] && row[1] < row[2])) return null;
    let label = `${row[0]}:${row[1]}–${row[2]}`;
    if (row.length === 6) {
      if (typeof row[3] !== 'string' || typeof row[4] !== 'number' || !['+', '-', '.'].includes(row[5])) return null;
      label += `; name: ${row[3]}; score: ${row[4]}; strand: ${row[5]}`;
    }
    labels.push(label);
  }
  return labels.length ? labels.join('; ') + ' (zero-based; end excluded)' : null;
}
function formatRecordValue(value, key = '') {
  value = reportParseValue(value, key);
  const position = reportBedPositions(value, key);
  if (position !== null) return position;
  if (Array.isArray(value)) return value.map(item => formatRecordValue(item, key)).join('; ') || 'None listed';
  if (value && typeof value === 'object') return Object.entries(value).map(([k,v]) =>
    `${reportFieldLabel(k)}: ${formatRecordValue(v, k)}`).join('; ') || 'No details recorded';
  return reportScalarLabel(value, key);
}
function renderRecordValue(value, key = '') {
  value = reportParseValue(value, key);
  const position = reportBedPositions(value, key);
  if (position !== null) return escapeHtml(position);
  if (Array.isArray(value)) return value.length ? '<ul class="reader-value-list">' +
    value.map(item => '<li>' + renderRecordValue(item, key) + '</li>').join('') + '</ul>' : 'None listed';
  if (value && typeof value === 'object') {
    const rows = Object.entries(value);
    if (!rows.length) return 'No details recorded';
    return '<dl class="reader-value-record">' + rows.map(([k,v]) => {
      let body = renderRecordValue(v, k);
      const nested = reportParseValue(v, k);
      if (nested && typeof nested === 'object' && Object.keys(nested).length && reportBedPositions(nested, k) === null) {
        const count = Object.keys(nested).length;
        const noun = Array.isArray(nested) ? 'item' : 'field';
        body = `<details class="reader-value-details"><summary>${count} ${noun}${count === 1 ? '' : 's'}</summary>${body}</details>`;
      }
      return '<div><dt>' + escapeHtml(reportFieldLabel(k)) + '</dt><dd>' + body + '</dd></div>';
    }).join('') + '</dl>';
  }
  return escapeHtml(reportScalarLabel(value, key));
}
'''
