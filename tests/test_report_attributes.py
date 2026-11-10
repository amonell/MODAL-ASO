"""Arbitrary saved evidence stays exact, typed, scoped and small in reports."""
from copy import deepcopy
import json

import pytest

from aso_agents.report_attributes import flatten_attributes, pack_attributes
from aso_agents.report_explorer_data import project_records
from aso_agents.report_explorer import explorer_html


def row(**extra):
    return {'antisense_5to3': 'ACGT', 'ordering_notation': 'ACGT', **extra}


def attribute(data, label):
    return next(a for a in data['attributes'] if a['label'] == label)


def unpack(attributes, count):
    catalog = {a['id']: a for a in attributes}
    restored = []
    for a in attributes:
        source = a
        while 'value_ref' in source:
            source = catalog[source['value_ref']]
        values = [source['constant']] * count if 'constant' in source else source['values']
        if 'dictionary' in source:
            values = [source['dictionary'][v] for v in values]
        if 'atoms' in source:
            decode = lambda v: None if v is None else source['atoms'][v]
            values = [[decode(v) for v in cell] if isinstance(cell, list) else decode(cell) for cell in values]
        restored.append({**{k: v for k, v in a.items() if k not in
                           {'value_ref', 'constant', 'dictionary', 'atoms', 'values'}}, 'values': values})
    return restored


def test_new_nested_fields_and_json_collections_are_available_without_registration():
    rows = [row(custom={'measurement': '0', 'replicates': [0, -2.75, None], 'flag': False,
                        'batch_id': '002', 'gene_ids': ['123', 'ENSG00001']},
                external_json='{"score": 12.3456789012345, "state": "reviewed"}'),
            row(custom={'measurement': '2.5', 'replicates': [], 'flag': True, 'batch_id': '003'})]
    before = deepcopy(rows)
    data = project_records(rows)
    assert attribute(data, 'Custom · Measurement')['values'] == [0, 2.5]
    replicates = attribute(data, 'Custom · Replicates')
    assert replicates['kind'] == 'number' and replicates['multiple']
    assert replicates['values'] == [[0, -2.75, None], None]
    assert attribute(data, 'Custom · Batch id')['values'] == ['002', '003']
    assert attribute(data, 'Custom · Gene ids')['values'] == [['123', 'ENSG00001'], None]
    assert attribute(data, 'Custom · Flag')['labels'] == {'false': 'No', 'true': 'Yes'}
    assert attribute(data, 'External json · Score')['values'] == [12.3456789012345, None]
    assert all('[]' not in a['label'] for a in data['attributes'])
    assert rows == before


def test_conservation_zero_negative_and_missing_coverage_remain_distinct():
    data = project_records([row(Phastcon=0, PhyloP=-2, pc_covered_bases=4, ph_covered_bases=4),
                            row(Phastcon=0, PhyloP=0, pc_covered_bases=0, ph_covered_bases=0),
                            row(phastcons=.9, phylop=3), row(Phastcon=float('nan'))])
    assert attribute(data, 'Conservation · phastCons')['values'] == [0, None, .9, None]
    assert attribute(data, 'Conservation · phyloP')['values'] == [-2, None, 3, None]
    json.dumps(data, allow_nan=False)


def test_gene_wide_coverage_and_isoform_selectivity_have_separate_attributes():
    data = project_records([row(isoform_uniqueness_score=.8, isoform_uniqueness_class='gene_wide'),
                            row(isoform_uniqueness_score=.5, isoform_uniqueness_class='target_shared_with_beneficial'),
                            row(isoform_uniqueness_score=0, isoform_uniqueness_class='non_target'),
                            row(isoform_uniqueness_score=1, isoform_uniqueness_class='unresolved_role_identity')])
    assert attribute(data, 'Isoform coverage')['values'] == [.8, None, None, None]
    assert attribute(data, 'Uniqueness · target isoform')['values'] == [None, .5, 0, None]


@pytest.mark.parametrize('bad', [dict(off_target_search_status='failed'), dict(off_target_mapping_capped=True),
                               dict(Off_target_2mm_count=None), dict(Off_target_2mm_count=-1),
                               dict(Off_target_2mm_count=.5)])
def test_uniqueness_uses_complete_comparable_integer_counts_including_three_mismatches(bad):
    source = row(Off_target_mismatch_budget=3, off_target_count_status='complete_within_declared_scope',
                 off_target_search_status='complete', Off_target_exact_count=0, Off_target_1mm_count=1,
                 Off_target_2mm_count=2, Off_target_3mm_count=3, off_target_exact_count_unit='transcripts',
                 Off_target_total_upto_mismatch_count=6)
    data = project_records([source, {**source, **bad}, {**source, 'Off_target_mismatch_budget': 1,
                                                    'Off_target_total_upto_mismatch_count': 1}])
    attributes = [a for a in data['attributes'] if a['id'].startswith('uniqueness')]
    by_budget = {3 if '3 mismatches' in a['label'] else 1: a['values'] for a in attributes}
    assert by_budget == {3: [6, None, None], 1: [None, None, 1]}


def test_uniqueness_never_double_counts_genes_across_mismatch_bins():
    source = row(Off_target_mismatch_budget=2, off_target_search_status='complete',
                 Off_target_exact_count=1, Off_target_1mm_count=1, Off_target_2mm_count=2,
                 off_target_exact_count_unit='genes', Off_target_total_upto_mismatch_count=2,
                 Off_target_gene_ids_by_mismatch={'0mm':['A'], '1mm':['A'], '2mm':['A','B']})
    data = project_records([source, {**source, 'Off_target_total_upto_mismatch_count':None},
                            {**source, 'Off_target_total_upto_mismatch_count':0}])
    assert attribute(data, 'Uniqueness · off-target matches')['values'] == [2, None, None]
    assert attribute(data, 'Off target exact count')['values'] == [1,1,1]


def test_large_references_and_paths_have_stable_readable_identities_and_stay_out_of_html():
    source = row(extra={'rna_context': 'ACGT' * 100000, 'reference_path': '/private/reference/genome.fasta'})
    data = project_records([source, deepcopy(source)])
    a = attribute(data, 'Extra · Rna context')
    assert a['values'][0] == a['values'][1] and a['values'][0].startswith('content:')
    assert list(a['labels'].values()) == ['RNA/reference sequence · 400,000 bases']
    document = explorer_html(data)
    assert 'ACGT' * 100 not in document and '/private/reference' not in document
    assert len(document) < 50000


def test_shared_columns_and_nested_atoms_pack_losslessly_without_rounding_or_type_coercion():
    values = [[f'GENE_IDENTIFIER_{(i + j) % 31:08}' for j in range(30)] for i in range(300)]
    attributes = [{'id': str(i), 'label': str(i), 'values': deepcopy(values)} for i in range(5)]
    attributes += [{'id': 'typed', 'values': [None, 0, False, '0', -1.234567890123, [True, None, 1]] * 50},
                   {'id': 'constant', 'values': [0] * 300},
                   {'id': 'precise', 'values': [i / 7 for i in range(300)]}]
    before = deepcopy(attributes)
    packed = pack_attributes(attributes)
    assert unpack(packed, 300) == attributes == before
    assert len(json.dumps(packed)) < len(json.dumps(attributes)) / 15
    assert any('atoms' in a for a in packed) and sum('value_ref' in a for a in packed) == 4


def test_only_complete_json_is_decoded_and_script_injection_stays_literal():
    source = row(custom='{"bad":', name='</script><img onerror="alert(1)">')
    flat = flatten_attributes(source)
    assert flat[('custom',)]['values'] == ['{"bad":']
    document = explorer_html(project_records([source]))
    assert '</script><img' not in document and '\\u003c/script>' in document
