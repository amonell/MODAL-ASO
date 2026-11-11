"""Offline, reference-bound exon naming for every report projection."""
from .curated_exon_nomenclature import reference_context, reference_fingerprint
import html
from urllib.parse import urlsplit


def ordinal_word(value):
    names = ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
             'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth',
             'eighteenth', 'nineteenth', 'twentieth')
    if type(value) is not int or value < 1:
        return ''
    if value <= len(names):
        return names[value-1]
    suffix = 'th' if 10 <= value % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th')
    return str(value)+suffix


def project_exon_names(entry):
    context = reference_context(entry)
    exons = entry.get('exons_tx') or []
    saved = (entry.get('metadata') or {}).get('curated_exon_nomenclature') or {}
    verified = (saved.get('status') == 'verified' and
        saved.get('reference_fingerprint') == reference_fingerprint(context['transcript_id'], exons, context) and
        saved.get('reference_transcript_id') == context['transcript_id'] and
        all(isinstance(saved.get(key), str) and len(saved[key]) == 64 and
            all(c in '0123456789abcdef' for c in saved[key]) for key in ('source_sha256', 'xref_sha256')) and
        bool(saved.get('evidence_refs')))
    names = {e['index']: e for e in saved.get('exons') or []} if verified else {}
    projected = []
    for exon in exons:
        name = names.get(exon['index']) or {}
        label = name.get('label') if all(name.get(k) == exon.get(k) for k in ('start', 'end')) else None
        projected.append({**{key: exon[key] for key in ('index', 'start', 'end')},
            'verified_exon_label': label, 'label': f'Exon {label}' if label else f'{ordinal_word(exon["index"])} annotated exon'})
    links = [{'label': saved.get('lrg_id') or 'Curated exon names', 'url': saved['source_url']}] if verified else []
    if verified:
        links += [{'label': saved['naming_system'], 'url': url} for url in saved.get('naming_refs') or []]
    named = sum(e['verified_exon_label'] is not None for e in projected)
    coverage = (f' Curated names cover {named} of {len(projected)} exons; the remaining labels show annotation order.'
                if verified and named < len(projected) else '')
    reference_label = context['transcript_id'] or 'the selected transcript'
    if reference_label.startswith('SNAPGENE::'):
        declared = (entry.get('metadata') or {}).get('declared_reference')
        reference_label = f'the supplied reference for {declared}' if declared else 'the supplied reference'
    return {'exons': projected,
            'exon_naming': {'verified': bool(verified), 'description': (
                f'Exon names are documented in {saved["lrg_id"]}. Coordinates match {context["transcript_id"]}.'+coverage if verified else
                f'Exons are counted in RNA order along {reference_label}. Curated names could not be verified for this reference.'),
                'links': links,
                'exons': [{'name': e['label'], 'annotation_order': ordinal_word(e['index']),
                           'coordinates': f'{context["chrom"]}:{e["start"]:,}–{e["end"]:,}'} for e in projected]}}


def bind_site_exon_name(site, entry, projected_exons):
    """A scored junction must match the same transcript, strand and boundary."""
    result = {'verified_exon_label': None, 'annotation_exon_index': None}
    if not entry or site.get('reference_binding_status') != 'resolved_catalog_identity':
        return result
    context = reference_context(entry)
    reference = site.get('reference_transcript_id')
    if reference != context['transcript_id']:
        return result
    kind = site.get('site_kind') or site.get('site')
    point = site.get(kind) or site
    chrom = str(point.get('chrom') or '').removeprefix('chr')
    strand = {'1': '+', '-1': '-'}.get(str(point.get('strand')), point.get('strand'))
    if kind not in {'acceptor', 'donor'} or strand != context['strand'] or chrom != str(context['chrom']).removeprefix('chr'):
        return result
    candidates = [e for e in projected_exons if e['index'] == site.get('exon_index')]
    if len(candidates) != 1:
        return result
    exon = candidates[0]
    if ((kind == 'acceptor' and exon['index'] == 1) or
            (kind == 'donor' and exon['index'] == len(projected_exons))):
        return result
    boundary = exon['start'] if (kind == 'acceptor') == (strand == '+') else exon['end']
    if point.get('position') != boundary:
        return result
    return {'annotation_exon_index': exon['index'], 'verified_exon_label': exon['verified_exon_label']}


def exon_naming_html(contexts):
    esc = lambda value: html.escape(str(value), quote=True)
    sections = []
    for index, context in enumerate(contexts):
        naming = context.get('exon_naming')
        if not naming:
            continue
        links = ' · '.join(f'<a href="{esc(link["url"])}" target="_blank" rel="noopener noreferrer">{esc(link["label"])}</a>'
            for link in naming['links'] if urlsplit(link['url']).scheme in {'http', 'https'})
        rows = ''.join('<tr>'+''.join('<td>'+esc(value)+'</td>' for value in
                       (exon['name'], exon['annotation_order'].capitalize(), exon['coordinates']))+'</tr>' for exon in naming['exons'])
        sections.append(f'<div data-exon-context="{esc(context["id"])}"'+(' hidden' if index else '')+'>'
            '<details class="report-details"><summary>Exon names and reference</summary><p>'+esc(naming['description'])+
            '</p>'+('<p>'+links+'</p>' if links else '')+'<div class="report-table-wrap"><table data-report-native="true"><thead><tr>'
            '<th>Exon</th><th>Order in this transcript</th><th>Coordinates</th></tr></thead><tbody>'+rows+
            '</tbody></table></div></details></div>')
    return ''.join(sections)
