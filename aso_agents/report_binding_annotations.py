"""Offline transcript and regulatory tracks projected from saved run evidence."""
import html
from urllib.parse import urlsplit

from .curated_exon_nomenclature import reference_context


FEATURES = {
    'utr5': ('5′ UTR', '#83b7cb'), 'cds': ('CDS', '#386777'),
    'utr3': ('3′ UTR', '#e7ae58'), 'noncoding': ('Noncoding exon', '#b9b0cf'),
    'unknown': ('CDS/UTR unannotated', '#c9d6dc'),
}
SITE_COLORS = {'rbp': '#7857a8', 'mirna': '#be526f'}


def _chrom(value):
    return str(value or '').removeprefix('chr')


def _assembly(value):
    return {'hg38': 'GRCh38', 'hg19': 'GRCh37', 'mm10': 'GRCm38', 'mm39': 'GRCm39'}.get(value, value)


def _same_reference(source, reference):
    if not source:
        return True
    # A versioned source must match exactly; a stored versionless ID may name
    # the resolved reference, but cannot authorize transferring another version.
    return source == reference or ('.' not in source and source == reference.split('.')[0])


def transcript_features(entry):
    """Partition each exon using this record's CDS, never its earlier lookup."""
    context = reference_context(entry or {})
    exons = (entry or {}).get('exons_tx') or []
    metadata = (entry or {}).get('metadata') or {}
    known = bool(exons) and context['strand'] in {'+', '-'}
    coding = []
    for exon in exons:
        lo, hi = exon.get('start'), exon.get('end')
        if type(lo) is not int or type(hi) is not int or not 1 <= lo <= hi:
            return {'status': 'unavailable', 'segments': [], 'note': 'Transcript coordinates are unavailable.'}
        a, b = exon.get('coding_start_genomic'), exon.get('coding_end_genomic')
        if exon.get('contains_coding') is True and type(a) is int and type(b) is int:
            a, b = sorted((a, b))
            if lo <= a <= b <= hi:
                coding.append((a, b))
            else:
                known = False
        elif exon.get('contains_coding') is not False or a is not None or b is not None:
            known = False
    # No CDS is not, by itself, evidence of a UTR or a noncoding transcript.
    noncoding = known and not coding and metadata.get('biotype') in {
        'lncRNA', 'lincRNA', 'antisense', 'processed_transcript', 'miRNA', 'snRNA', 'snoRNA', 'rRNA'}
    bounds = (min(a for a, _ in coding), max(b for _, b in coding)) if coding and known else None
    segments = []
    for exon in exons:
        lo, hi = exon['start'], exon['end']
        pieces = [(lo, hi, 'noncoding' if noncoding else 'unknown')]
        if bounds:
            pieces = []
            cuts = {lo, hi + 1}
            for point in (bounds[0], bounds[1] + 1):
                if type(point) is int and lo < point <= hi:
                    cuts.add(point)
            if exon.get('contains_coding'):
                a, b = sorted((exon['coding_start_genomic'], exon['coding_end_genomic']))
                cuts.update(point for point in (a, b + 1) if lo < point <= hi)
            edges = sorted(cuts)
            for start, stop in zip(edges, edges[1:]):
                if stop - 1 < bounds[0]:
                    kind = 'utr5' if context['strand'] == '+' else 'utr3'
                elif start > bounds[1]:
                    kind = 'utr3' if context['strand'] == '+' else 'utr5'
                elif exon.get('contains_coding') and a <= start <= stop - 1 <= b:
                    kind = 'cds'
                else:
                    kind = 'unknown'
                pieces.append((start, stop - 1, kind))
        segments.extend({'chrom': context['chrom'], 'start': a, 'end': b,
                         'exon_index': exon['index'], 'kind': kind, 'label': FEATURES[kind][0]}
                        for a, b, kind in pieces)
    resolved = bool(segments) and all(s['kind'] != 'unknown' for s in segments)
    return {'status': 'resolved' if resolved else 'unavailable', 'segments': segments,
            'note': ('CDS and UTR boundaries use the saved transcript annotation.' if bounds and resolved else
                     'The saved annotation identifies a noncoding transcript.' if noncoding else
                     'CDS/UTR annotation is incomplete; unannotated sequence is shown in gray.')}


def regulatory_sites(entry, materialized, program_id):
    """Use only materialized genomic sites in the selected program/reference."""
    context = reference_context(entry or {})
    constraints = [*(materialized.get('global') or []),
                   *((materialized.get('by_program') or {}).get(program_id) or [])]
    sites, notes, seen = [], [], set()
    for constraint in constraints:
        source_type = constraint.get('source_type')
        if source_type not in {'rbp_binding', 'mirna_binding'}:
            continue
        summary = constraint.get('materialization_summary') or {}
        dataset = summary.get('selected_dataset') or {}
        scope = summary.get('feature_scope') or {}
        assembly = dataset.get('assembly') or summary.get('assembly')
        reference = context['transcript_id']
        if (constraint.get('coordinate_space') != 'genomic' or not context['chrom'] or
                context['strand'] not in {'+', '-'} or not reference or
                not _same_reference(scope.get('reference_transcript'), reference) or
                assembly and context['assembly'] and _assembly(assembly) != _assembly(context['assembly'])):
            notes.append('Some regulatory sites could not be matched to this transcript reference and are not plotted.')
            continue
        records = summary.get('site_records') or []
        for row in constraint.get('materialized_rows') or []:
            if (not isinstance(row, (list, tuple)) or len(row) < 6 or
                    type(row[1]) is not int or type(row[2]) is not int or not 0 <= row[1] < row[2] or
                    _chrom(row[0]) != _chrom(context['chrom']) or row[5] != context['strand']):
                notes.append('Some regulatory sites lack matching genomic coordinates or strand and are not plotted.')
                continue
            kind = 'rbp' if source_type == 'rbp_binding' else 'mirna'
            record = next((r for r in records if any(list(s[:3]) == list(row[:3])
                          for s in r.get('genomic_segments') or [])), {})
            identity = record.get('reference_identity') or {}
            site_reference = identity.get('versioned_id') or identity.get('id')
            if (not _same_reference(site_reference, reference) or
                    identity.get('assembly_name') and context['assembly'] and
                    _assembly(identity['assembly_name']) != _assembly(context['assembly']) or
                    (record.get('sequence_projection') or {}).get('status') not in {None, 'verified'}):
                notes.append('Some miRNA sites have unresolved or conflicting reference projections and are not plotted.')
                continue
            if kind == 'rbp':
                label = summary.get('rbp') or (constraint.get('rbp_binding') or {}).get('rbp') or 'RBP'
                assay = summary.get('assay') or dataset.get('assay_title')
                evidence = f'{assay} peak' if assay in {'eCLIP', 'CLIP', 'iCLIP', 'PAR-CLIP', 'HITS-CLIP'} else 'Recorded RBP site'
                sample = dataset.get('biosample_term_name') or summary.get('biosample_hint')
                source = ' · '.join(str(v) for v in (sample, dataset.get('file_accession')) if v)
                url = dataset.get('download_url')
            else:
                label = record.get('family') or '/'.join(summary.get('resolved_families') or summary.get('requested_mirnas') or []) or 'miRNA'
                source = ' · '.join(str(v) for v in (summary.get('source'), summary.get('release')) if v)
                evidence = ('Predicted ' + str(record.get('site_type') or 'binding') + ' site'
                            if 'targetscan' in source.lower() else 'Recorded miRNA site')
                url = None
            key = (kind, str(label), *row[:3], str(source), str(evidence))
            if key in seen:
                continue
            seen.add(key)
            sites.append({'kind': kind, 'label': str(label), 'chrom': str(row[0]),
                          'start': row[1] + 1, 'end': row[2], 'strand': row[5],
                          'source_start0': row[1], 'source_end': row[2],
                          'evidence': evidence, 'source': source or 'Source details unavailable',
                          'url': url, 'reference_transcript': reference,
                          'constraint_id': constraint.get('constraint_id')})
        if not constraint.get('materialized_rows'):
            notes.append('No materialized genomic sites were saved for ' + str(constraint.get('label') or source_type) + '.')
    return {'regulatory_sites': sites, 'binding_annotation_notes': list(dict.fromkeys(notes))}


def annotation_html(contexts):
    """An RNA-order overview and accessible, source-labelled regulatory table."""
    esc = lambda value: html.escape(str(value or ''), quote=True)
    parts = []
    for index, context in enumerate(contexts):
        features = context.get('reference_features') or {}
        segments = features.get('segments') or []
        sites = context.get('regulatory_sites') or []
        exons = sorted(context.get('exons') or [], key=lambda e: e['index'])
        parts.append('<div data-binding-context="' + esc(context['id']) + '"' + (' hidden' if index else '') + '>')
        if exons and segments:
            width, gap, left = 800, min(24, 160 / max(1, len(exons) - 1)), 22
            total = sum(e['end'] - e['start'] + 1 for e in exons)
            available = width - left * 2 - gap * (len(exons) - 1)
            overview = ['<svg viewBox="0 0 800 91" role="img" aria-label="Transcript overview in RNA order, 5′ to 3′" '
                        'style="display:block;width:100%;height:auto;min-width:300px">',
                        '<text x="4" y="31" font-size="11" fill="#536673">5′</text>',
                        '<text x="795" y="31" text-anchor="end" font-size="11" fill="#536673">3′</text>']
            offset = left
            for exon in exons:
                size = (exon['end'] - exon['start'] + 1) / total * available
                if offset > left:
                    overview.append(f'<path d="M{offset-gap} 27H{offset}" stroke="#b5c2c8" stroke-dasharray="3 3"/>')
                for segment in (s for s in segments if s['exon_index'] == exon['index']):
                    delta = (segment['start'] - exon['start'] if context.get('reference_strand') != '-' else exon['end'] - segment['end'])
                    x = offset + delta / total * available
                    w = (segment['end'] - segment['start'] + 1) / total * available
                    label, color = FEATURES[segment['kind']]
                    height = 20 if segment['kind'] == 'cds' else 12
                    title = f'{exon["label"]} · {label} · {segment["chrom"]}:{segment["start"]:,}–{segment["end"]:,}'
                    overview.append(f'<rect x="{x:.3f}" y="{27-height/2}" width="{max(.5,w):.3f}" height="{height}" fill="{color}"><title>{esc(title)}</title></rect>')
                    if w > len(label) * 7 + 16:
                        overview.append(f'<text x="{x+w/2:.3f}" y="16" text-anchor="middle" font-size="11" fill="#354752">{esc(label)}</text>')
                short = exon.get('verified_exon_label')
                label = f'Exon {short}' if short else f'Exon {exon["index"]} in RNA order'
                if size > len(label) * 6 + 6:
                    overview.append(f'<text x="{offset+size/2:.3f}" y="54" text-anchor="middle" font-size="11" fill="#536673">{esc(label)}</text>')
                # Regulatory sites are small ticks here; exact spans are below.
                for site in sites:
                    a, b = max(site['start'], exon['start']), min(site['end'], exon['end'])
                    if b < a:
                        continue
                    delta = a - exon['start'] if context.get('reference_strand') != '-' else exon['end'] - b
                    x = offset + delta / total * available
                    overview.append(f'<rect x="{x:.3f}" y="65" width="{max(2,(b-a+1)/total*available):.3f}" height="5" fill="{SITE_COLORS[site["kind"]]}"><title>{esc(site["label"]+" · "+site["evidence"])}</title></rect>')
                offset += size + gap
            overview.append('</svg>')
            parts.append('<p class="report-sub"><strong>Transcript overview · ' + esc(context.get('transcript')) + '</strong><br>RNA order 5′→3′; introns compressed. The binding map below uses genomic coordinates.</p><div style="overflow-x:auto">' + ''.join(overview) + '</div>')
            present = list(dict.fromkeys(s['kind'] for s in segments))
            parts.append('<div class="report-legend"><div class="report-category-legend">' + ''.join(
                f'<span><i style="background:{FEATURES[k][1]}"></i>{FEATURES[k][0]}</span>' for k in present) + '</div></div>')
            parts.append('<p class="report-sub">' + esc(features.get('note')) + '</p>')
        if sites:
            source_notes = list(dict.fromkeys(s['label'] + ' · ' + s['evidence'] + ' · ' + s['source'] for s in sites))
            parts.append('<div class="report-legend"><div class="report-category-legend">' + ''.join(
                f'<span><i style="background:{SITE_COLORS[k]}"></i>{label}</span>'
                for k, label in [('rbp', 'RBP binding regions'), ('mirna', 'miRNA binding sites')] if any(s['kind'] == k for s in sites)) + '</div></div>')
            summary = '; '.join(source_notes[:4])
            if len(source_notes) > 4:
                summary += f'; {len(source_notes)-4} more source/target combinations in the evidence table'
            parts.append('<p class="report-sub">' + esc('Sources: ' + summary) + '.</p>')
            parts.append('<details class="report-map-sites"><summary>Regulatory sites and evidence · ' + str(len(sites)) + '</summary><div class="report-map-site-list"><table data-report-native="true"><thead><tr><th>Target</th><th>Genomic interval (1-based)</th><th>Evidence</th><th>Source</th></tr></thead><tbody>')
            for site in sites:
                source = esc(site['source'])
                if urlsplit(site.get('url') or '').scheme in {'http', 'https'}:
                    source = '<a href="' + esc(site['url']) + '" target="_blank" rel="noopener noreferrer">' + source + '</a>'
                parts.append('<tr><td>' + esc(site['label']) + '</td><td>' + esc(f'{site["chrom"]}:{site["start"]:,}–{site["end"]:,} ({site["strand"]})') + '</td><td>' + esc(site['evidence']) + '</td><td>' + source + '</td></tr>')
            parts.append('</tbody></table></div></details>')
        for note in context.get('binding_annotation_notes') or []:
            parts.append('<p class="report-sub">' + esc(note) + '</p>')
        parts.append('</div>')
    return ''.join(parts)
