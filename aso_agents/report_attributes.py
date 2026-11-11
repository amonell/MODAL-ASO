"""Typed, compact attribute columns over saved ASO evidence. No new analysis."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from .report_values import field_label, scalar_label


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _scalar(value, path):
    if isinstance(value, str):
        # CSV-derived measurements can be numeric strings. Identity fields keep
        # their literal text, including numeric transcript/chromosome labels.
        key = next((part.lower() for part in reversed(path) if part!='[]'), '')
        identity = key.endswith(('id', 'ids', 'name', 'names', 'label', 'labels', 'sequence', 'sequences',
                                 'chrom', 'strand', 'identifier', 'identifiers', 'hash', 'sha256', 'sha1', 'checksum'))
        text = value.strip()
        if not identity and text and text[0] in '-0123456789':
            try:
                parsed = json.loads(value)
                if _finite(parsed):
                    return parsed
            except (ValueError, TypeError):
                pass
        return value
    return value if isinstance(value, bool) or _finite(value) else None


def flatten_attributes(row):
    """Keep scalar leaves and collections; repeated records share one field.

    Large reference sequences and file paths are represented by stable identity
    and a readable label. Their complete content stays in the evidence files.
    """
    source = {**(row.get('source_annotations') or {}), **row}
    source.pop('source_annotations', None)
    fields = defaultdict(list)
    labels = defaultdict(dict)

    def visit(value, path):
        if isinstance(value, str) and value.lstrip().startswith(('[', '{')):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, dict)):
                    value = parsed
            except ValueError:
                pass
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, (*path, str(key)))
        elif isinstance(value, (list, tuple)):
            if not value:
                fields[(*path, '[]')].append(None)
            for item in value:
                visit(item, (*path, '[]'))
        else:
            value = _scalar(value, path)
            if isinstance(value, str):
                key = next((part.lower() for part in reversed(path) if part != '[]'), '')
                reference_sequence = key in {'rna_context', 'reference_sequence', 'genomic_sequence', 'genome_sequence', 'reference_rna_sequence'}
                file_path = value.startswith(('/', 'file://')) or key.endswith(('_path', '_file', '_fasta'))
                if reference_sequence or file_path or len(value)>2048:
                    token = 'content:' + hashlib.sha256(value.encode()).hexdigest()
                    label = (f'RNA/reference sequence · {len(value):,} bases' if reference_sequence else
                             Path(value).name if file_path else f'Long text · {len(value):,} characters')
                    labels[path][token] = label
                    value = token
            fields[path].append(value)
    visit(source, ())
    return {path: {'values': values, 'labels': labels.get(path, {})}
            for path, values in fields.items() if path}


def _label(path):
    # The collection marker explains scope in metadata, never as code notation.
    return ' · '.join(field_label(key).replace('@', ' at ') for key in path if key!='[]')


def is_color_identity(path, labels=()):
    """Identify per-record text and references that carry no color scale."""
    if any(key.startswith('content:') for key in labels):
        return True
    key = next((part for part in reversed(path) if part != '[]'), '')
    key = '_'.join(key.lower().replace('-', ' ').replace('_', ' ').split())
    return key in {
        'id', 'name', 'aso', 'candidate', 'idt', 'synthesis', 'sequence',
        'aso_name', 'candidate_name', 'record_name', 'candidate_number',
        'aso_id', 'candidate_id', 'molecule_id', 'record_id', 'tile_id',
        'molecule_identity', 'input_id', 'row_index', 'tile_index',
        'rna_context', 'antisense_5to3', 'antisense_5_to_3',
        'synthesized_antisense_sequence', 'target_5to3', 'ordering_notation', 'notation',
    } or key.endswith(('_sequence', '_sequences', '_notation',
                       '_sha256', '_sha1', '_hash', '_checksum', '_path', '_file'))


def _number(source, *keys):
    for key in keys:
        if key in source and source[key] is not None:
            value = _scalar(source[key], (key,))
            return value if _finite(value) else None
    return None


def common_attributes(row, record):
    source = {**(row.get('source_annotations') or {}), **row}
    score = _number(source, 'isoform_uniqueness_score')
    kind = source.get('isoform_uniqueness_class')
    selective = {'target_only', 'target_without_beneficial', 'target_shared_with_beneficial', 'non_target'}
    phastcons = _number(source, 'Phastcon', 'phastcons', 'phastcon')
    phylop = _number(source, 'PhyloP', 'phylop', 'phyloP')
    if _number(source, 'pc_covered_bases') == 0:
        phastcons = None
    if _number(source, 'ph_covered_bases') == 0:
        phylop = None
    screen = (row.get('candidate_assessment') or row.get('assessment') or {}).get('off_target') or {}
    counts = screen.get('counts_by_mismatch')
    if not isinstance(counts, list):
        counts = [_number(source, 'Off_target_' + key + '_count') for key in ('exact', '1mm', '2mm', '3mm')]
    counts = [_scalar(value, ('count',)) for value in counts]
    total = _number(source, 'Off_target_total_upto_mismatch_count', 'off_target_total_upto_mismatch_count')
    source_budget = _number(source, 'Off_target_mismatch_budget')
    if source_budget is not None and source_budget != record.get('off_target_budget'):
        total = None
    return {
        'effect': record['effect'], 'gc': record['gc'] * 100 if record['gc'] is not None else None,
        'phastcons': phastcons, 'phylop': phylop,
        'isoform_selectivity': score if kind in selective else None,
        'isoform_coverage': score if kind=='gene_wide' else _number(source, 'isoform_coverage_fraction'),
        'native_rank': record['native_rank'], 'length':len(record['aso']),
        'off_target_counts': counts,
        'off_target_total': total,
    }


COMMON = [
    ('oligoai', 'OligoAI RNA reduction', 'Predicted RNA reduction using the selected RNA contexts, dose and delivery. Colors and filters use the lowest available prediction for each ASO; incomplete predictions remain unavailable.', '%', 0, 100, 'higher'),
    ('effect', 'Splice prediction', 'Predicted support for the requested splicing direction at this target.', '', None, None, 'higher'),
    ('phastcons', 'Conservation · phastCons', 'Average conservation across the binding site. Higher values indicate stronger evolutionary conservation.', '', 0, 1, 'higher'),
    ('phylop', 'Conservation · phyloP', 'Average phyloP across the binding site. Positive values indicate slower evolution; negative values indicate faster evolution.', '', None, None, 'higher'),
    ('isoform_selectivity', 'Uniqueness · target isoform', 'Saved selectivity for the requested isoform. Higher values indicate more selective targeting; matches to protected isoforms carry the recorded penalty.', '', 0, 1, 'higher'),
    ('isoform_coverage', 'Isoform coverage', 'Fraction of the loaded transcript catalog containing the binding site.', '', 0, 1, 'higher'),
    ('gc', 'GC content', 'Percentage of G and C bases in the ASO.', '%', 0, 100, 'higher'),
    ('length', 'ASO length', 'Length of the synthesized ASO.', 'nt', None, None, 'higher'),
    ('native_rank', 'Analysis rank', 'Rank from the saved analysis within this target and approach.', '', None, None, 'lower'),
]


def attach_attribute_catalog(result):
    """Move per-row evidence into shared columns so metadata is stored once."""
    records = result['records']
    raw = [record.pop('_raw_attributes', {}) for record in records]
    common = [record.pop('_common_attributes', {}) for record in records]
    attributes = []
    for key, label, description, unit, low, high, direction in COMMON:
        values = [item.get(key) for item in common]
        if key=='oligoai' and not any(r.get('oligoai_applicable') or r.get('oligoai') for r in records):
            continue
        if key!='oligoai' and not any(v is not None for v in values):
            continue
        attributes.append({'id':key, 'label':label, 'group':'Common attributes', 'kind':'number',
                           'description':description, 'unit':unit, 'low':low, 'high':high, 'direction':direction,
                           'values':values, **({'dynamic':'oligoai'} if key=='oligoai' else {})})
    # Uniqueness comparisons retain their mismatch budget and counting unit.
    scopes = sorted({(r.get('off_target_budget'),r.get('off_target_unit')) for r in records
                     if r.get('off_target_complete') and r.get('off_target_budget') in (0,1,2,3)}, key=str)
    for budget,unit in scopes:
        values=[]
        for record, saved in zip(records, common):
            counts=saved.get('off_target_counts',[])[:int(budget)+1]
            valid=(record.get('off_target_complete') and record.get('off_target_budget')==budget
                   and record.get('off_target_unit')==unit and len(counts)==budget+1
                   and all(_finite(v) and v>=0 and float(v).is_integer() for v in counts))
            total = saved.get('off_target_total')
            # Genes/transcripts can occur in several mismatch bins. A saved
            # union count preserves their identity across those bins. Locus
            # bins, and a single occupied bin, have an unambiguous sum.
            if valid and total is None and (unit=='loci' or sum(v>0 for v in counts)<=1):
                total = sum(counts)
            valid = (valid and _finite(total) and float(total).is_integer()
                     and max(counts, default=0)<=total<=sum(counts))
            values.append(total if valid else None)
        displayed_unit = {'loci':'genomic positions', 'unresolved':'recorded matches',
                          'source_defined_hits; do_not_interpret_as_distinct_genes':'recorded matches'}.get(unit,unit)
        key='uniqueness' if len(scopes)==1 else 'uniqueness_'+hashlib.sha256(str((budget,unit)).encode()).hexdigest()[:12]
        attributes.insert(1, {'id':key,'label':'Uniqueness · off-target matches'+(f' · {budget:g} mismatches · {unit}' if len(scopes)>1 else ''),
            'group':'Common attributes','kind':'number','direction':'lower','low':0,'high':None,'unit':displayed_unit,
            'description':f'Other {displayed_unit} matching with up to {int(budget)} mismatches. Fewer matches appear darker. Counts use completed searches with this limit.',
            'values':values})
    paths=sorted({path for item in raw for path in item})
    for path in paths:
        columns=[item.get(path,{}).get('values',[None]) for item in raw]
        values=[column[0] if len(column)==1 else column for column in columns]
        valid=[v for column in columns for v in column if v is not None]
        if not valid and path[-1]=='[]' and any(other[:len(path)]==path and len(other)>len(path) for other in paths):
            continue  # Empty containers contribute missing values to their leaves.
        numeric=bool(valid) and all(_finite(v) for v in valid)
        labels={key:label for item in raw for key,label in item.get(path,{}).get('labels',{}).items()}
        # Distinct protected payloads with the same short label remain distinct.
        by_label=defaultdict(list)
        for key,label in labels.items(): by_label[label].append(key)
        for label,keys in by_label.items():
            if len(keys)>1:
                for i,key in enumerate(sorted(keys),1): labels[key]=f'{label} · item {i}'
        if not numeric:
            seen_labels = set(labels)
            for value in valid:
                key = str(value).lower() if isinstance(value, bool) else str(value)
                if key not in seen_labels:
                    seen_labels.add(key)
                    caption = str(scalar_label(value, path[-1]))
                    if caption != key:
                        labels[key] = caption
        attributes.append({'id':'saved_'+hashlib.sha256(json.dumps(path).encode()).hexdigest()[:20],
            'label':_label(path),'group':'Saved attributes','kind':'number' if numeric else 'category',
            'description':('Values across saved records for each ASO.' if '[]' in path else 'Value recorded with each ASO.')
                          + (' Full reference content stays in the evidence files.' if any(k.startswith('content:') for k in labels) else ''),
            'multiple':any(len(column)>1 for column in columns), 'unit':'', 'direction':'higher',
            'values':values, **({'labels':labels} if labels else {}),
            **({'color_identity':True} if is_color_identity(path, labels) else {})})
    result['attributes']=attributes
    return result


def pack_attributes(attributes):
    """Losslessly share columns and repeated atoms in large evidence collections.

    Each option is measured, so a small or unusual attribute cannot inflate just
    to use a particular encoding. Metadata stays readable and fields stay
    independently selectable even when several saved paths share their values.
    """
    packed, shared = [], {}
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

    def compact(values):
        unique, lookup, indices = [], {}, []
        for value in values:
            key = encode(value)
            if key not in lookup:
                lookup[key] = len(unique)
                unique.append(value)
            indices.append(lookup[key])
        if len(unique) == 1:
            return {'constant': unique[0]}
        options = [{'values': values}, {'dictionary': unique, 'values': indices}]
        return min(options, key=lambda option: len(encode(option)))

    for attribute in attributes:
        meta={key:value for key,value in attribute.items() if key!='values'}
        values=attribute.get('values',[])
        serialized = encode(values)
        fingerprint = hashlib.sha256(serialized.encode()).digest()
        if fingerprint in shared:
            meta['value_ref'] = shared[fingerprint]
            packed.append(meta)
            continue
        shared[fingerprint] = attribute['id']
        options = [compact(values)]
        atoms, lookup = [], {}
        def atom(value):
            if value is None:
                return None
            key = encode(value)
            if key not in lookup:
                lookup[key] = len(atoms)
                atoms.append(value)
            return lookup[key]
        cells=[]
        for value in values:
            cells.append([atom(v) for v in value] if isinstance(value,list) else atom(value))
        options.append({**compact(cells), 'atoms': atoms})
        meta.update(min(options, key=lambda option: len(encode(option))))
        packed.append(meta)
    return packed
