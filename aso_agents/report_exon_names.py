"""Offline, reference-bound exon naming for every report projection."""
from .curated_exon_nomenclature import reference_context, reference_fingerprint, verified_mapping_for_entry
import html
from copy import deepcopy
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


def report_reference_catalog(goal):
    """Project saved typed feature locations for older and resumed reports."""
    from .exon_nomenclature_context import resolved_exon_requests, selected_reference
    catalog = deepcopy(goal.get('isoform_transcripts') or {})
    for feature in resolved_exon_requests(goal):
        if feature['status'] == 'unresolved' or feature.get('requested_exon_label') is None:
            continue
        _, entry = selected_reference(catalog, feature.get('reference_transcript_id'))
        if entry is None:
            continue
        context = reference_context(entry)
        identity = {**feature, 'reference_fingerprint': reference_fingerprint(
            context['transcript_id'], entry.get('exons_tx') or [], context)}
        entry.setdefault('metadata', {}).setdefault('execution_exon_identities', []).append(identity)
    return catalog


def project_exon_names(entry):
    context = reference_context(entry)
    exons = [{**exon, 'index': exon.get('index', index)}
             for index, exon in enumerate(entry.get('exons_tx') or [], 1)
             if isinstance(exon, dict) and type(exon.get('start')) is int and type(exon.get('end')) is int
             and 1 <= exon['start'] <= exon['end']]
    saved = (entry.get('metadata') or {}).get('curated_exon_nomenclature') or {}
    verified = (saved.get('status') == 'verified' and
        saved.get('reference_fingerprint') == reference_fingerprint(context['transcript_id'], exons, context) and
        saved.get('reference_transcript_id') == context['transcript_id'] and
        all(isinstance(saved.get(key), str) and len(saved[key]) == 64 and
            all(c in '0123456789abcdef' for c in saved[key]) for key in ('source_sha256', 'xref_sha256')) and
        bool(saved.get('evidence_refs')))
    names = {e['index']: e for e in saved.get('exons') or []} if verified else {}
    fingerprint = reference_fingerprint(context['transcript_id'], exons, context)
    mappings = (entry.get('metadata') or {}).get('exon_nomenclature_resolutions') or entry.get('exon_nomenclature_resolutions') or {}
    source_names = {}
    source_links = []
    for requested, resolution in mappings.items():
        if (verified_mapping_for_entry(resolution, entry) and resolution.get('reference_fingerprint') == fingerprint
                and resolution.get('evidence_refs')):
            source_names.setdefault(resolution['resolved_transcript_ordinal'], set()).add(str(requested))
            source_links.extend(resolution['evidence_refs'])
    identities = [i for i in (entry.get('metadata') or {}).get('execution_exon_identities') or []
                  if i.get('reference_fingerprint') == fingerprint and i.get('reference_transcript_id') == context['transcript_id']
                  and i.get('status') != 'unresolved']
    projected = []
    for exon in exons:
        name = names.get(exon['index']) or {}
        label = name.get('label') if all(name.get(k) == exon.get(k) for k in ('start', 'end')) else None
        extra = source_names.get(exon['index']) or set()
        if label is None and len(extra) == 1:
            label = next(iter(extra))
        requested = {str(i['requested_exon_label']) for i in identities if i.get('requested_exon_label') is not None
                     and (i.get('start'), i.get('end'), i.get('resolved_transcript_ordinal')) == (exon['start'], exon['end'], exon['index'])}
        requested_name = next(iter(requested)) if label is None and len(requested) == 1 else None
        supported = {str(i['requested_exon_label']) for i in identities if i.get('biological_label_verified')
                     and i.get('evidence_refs') and i.get('resolved_transcript_ordinal') == exon['index']
                     and (i.get('start'), i.get('end')) == (exon['start'], exon['end'])}
        if label is None and len(supported) == 1:
            label, requested_name = next(iter(supported)), None
        projected.append({**{key: exon[key] for key in ('index', 'start', 'end')},
            'verified_exon_label': label, 'requested_exon_label': requested_name,
            'label': f'Exon {label}' if label is not None else f'Requested exon {requested_name}' if requested_name else f'{ordinal_word(exon["index"])} annotated exon'})
    links = [{'label': saved.get('lrg_id') or 'Curated exon names', 'url': saved['source_url']}] if verified else []
    if verified:
        links += [{'label': saved['naming_system'], 'url': url} for url in saved.get('naming_refs') or []]
    links += [{'label': 'Exon name source', 'url': url} for url in dict.fromkeys(source_links)
              if urlsplit(url).scheme in {'http', 'https'} and url not in {link['url'] for link in links}]
    named = sum(e['verified_exon_label'] is not None for e in projected)
    coverage = (f' Curated names cover {named} of {len(projected)} exons; the remaining labels show annotation order.'
                if verified and named < len(projected) else '')
    reference_label = context['transcript_id'] or 'the selected transcript'
    if reference_label.startswith('SNAPGENE::'):
        declared = (entry.get('metadata') or {}).get('declared_reference')
        reference_label = f'the supplied reference for {declared}' if declared else 'the supplied reference'
    description = (
                f'Exon names are documented in {saved["lrg_id"]}. Coordinates match {context["transcript_id"]}.'+coverage if verified else
                f'{named} exon names have saved source support for {reference_label}. Other exons are counted in RNA order.' if named else
                f'Exons are counted in RNA order along {reference_label}. Curated names could not be verified for this reference.')
    if any(e['requested_exon_label'] for e in projected):
        description += ' Requested exon labels follow the physical locations confirmed for this run.'
    return {'exons': projected,
            'exon_naming': {'verified': bool(verified or named), 'description': description,
                'links': links,
                'exons': [{'name': e['label'], 'annotation_order': ordinal_word(e['index']),
                           'coordinates': f'{context["chrom"]}:{e["start"]:,}–{e["end"]:,}'} for e in projected]}}


def bind_site_exon_name(site, entry, projected_exons):
    """A scored junction must match the same transcript, strand and boundary."""
    result = {'verified_exon_label': None, 'annotation_exon_index': None, 'requested_exon_label': None}
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
    return {'annotation_exon_index': exon['index'], 'verified_exon_label': exon['verified_exon_label'],
            'requested_exon_label': exon.get('requested_exon_label')}


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
