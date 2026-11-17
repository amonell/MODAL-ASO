"""Reference-bound, offline annotations for shared report maps."""
from copy import deepcopy
import json

import pytest

from aso_agents.report_binding_annotations import annotation_html, regulatory_sites, transcript_features
from aso_agents.report_exon_names import project_exon_names
from aso_agents.report_explorer_data import project_run
from test_report_explorer import run_fixture, save


def reference(negative=False):
    exons = [
        {'index': 1, 'start': 100, 'end': 199, 'contains_coding': True,
         'coding_start_genomic': 150, 'coding_end_genomic': 199},
        {'index': 2, 'start': 300, 'end': 399, 'contains_coding': True,
         'coding_start_genomic': 300, 'coding_end_genomic': 349},
        {'index': 3, 'start': 500, 'end': 599, 'contains_coding': False,
         'coding_start_genomic': None, 'coding_end_genomic': None}]
    if negative:
        for e in exons:
            e['start'], e['end'] = 700 - e['end'], 700 - e['start']
            for k in ('coding_start_genomic', 'coding_end_genomic'):
                if e[k] is not None:
                    e[k] = 700 - e[k]
    return {'transcript_id': 'TX', 'chrom': 'chr1', 'strand': -1 if negative else 1,
            'metadata': {'versioned_id': 'TX.1', 'assembly_name': 'GRCh38', 'biotype': 'protein_coding'},
            'exons_tx': exons}


def constraints():
    return {'by_program': {'p': [
        {'source_type': 'rbp_binding', 'coordinate_space': 'genomic', 'constraint_id': 'rbp',
         'materialized_rows': [['chr1', 510, 530, 'peak', 1000, '+']],
         'materialization_summary': {'rbp': 'TEST-RBP', 'assay': 'eCLIP', 'assembly': 'hg38',
             'feature_scope': {'reference_transcript': 'TX'},
             'selected_dataset': {'biosample_term_name': 'K562', 'file_accession': 'ENCFFTEST'}}},
        {'source_type': 'mirna_binding', 'coordinate_space': 'genomic', 'constraint_id': 'miRNA',
         'materialized_rows': [['chr1', 545, 553, 'seed', .7, '+']],
         'materialization_summary': {'source': 'TargetScan', 'release': 'vert_80',
             'site_records': [{'family': 'miR-test', 'site_type': '8mer',
                 'reference_identity': {'versioned_id': 'TX.1', 'assembly_name': 'GRCh38'},
                 'sequence_projection': {'status': 'verified'},
                 'genomic_segments': [['chr1', 545, 553, 'seed', .7, '+']]}]}}
    ]}}


@pytest.mark.parametrize('negative', [False, True])
def test_cds_and_utr_partition_inclusive_coordinates_on_both_strands(negative):
    entry = reference(negative); before = deepcopy(entry)
    result = transcript_features(entry)
    assert result['status'] == 'resolved'
    expected = [(100, 149, 'utr5'), (150, 199, 'cds'), (300, 349, 'cds'),
                (350, 399, 'utr3'), (500, 599, 'utr3')]
    if negative:
        expected = [(700-b, 700-a, k) for a, b, k in expected]
    assert sorted((s['start'], s['end'], s['kind']) for s in result['segments']) == sorted(expected)
    assert sum(s['end']-s['start']+1 for s in result['segments']) == 300
    assert entry == before


@pytest.mark.parametrize('damage', ['missing_cds', 'invalid_cds', 'unknown_strand', 'no_cds'])
def test_missing_or_contradictory_annotation_does_not_invent_utr(damage):
    entry = reference()
    if damage == 'missing_cds':
        entry['exons_tx'][0].pop('coding_end_genomic')
    elif damage == 'invalid_cds':
        entry['exons_tx'][0]['coding_end_genomic'] = 999
    elif damage == 'unknown_strand':
        entry['strand'] = None
    else:
        for e in entry['exons_tx']:
            e.update(contains_coding=False, coding_start_genomic=None, coding_end_genomic=None)
    result = transcript_features(entry)
    assert result['status'] == 'unavailable'
    assert {s['kind'] for s in result['segments']} == {'unknown'}


def test_saved_noncoding_annotation_is_not_called_utr():
    entry = reference();entry['metadata']['biotype'] = 'lncRNA'
    for e in entry['exons_tx']:
        e.update(contains_coding=False, coding_start_genomic=None, coding_end_genomic=None)
    assert {s['kind'] for s in transcript_features(entry)['segments']} == {'noncoding'}


def test_regulatory_tracks_preserve_source_and_program_scope():
    entry, evidence = reference(), constraints();before = deepcopy(evidence)
    result = regulatory_sites(entry, evidence, 'p')
    rbp, mirna = result['regulatory_sites']
    assert (rbp['start'], rbp['end'], rbp['source_start0']) == (511, 530, 510)
    assert rbp['evidence'] == 'eCLIP peak' and rbp['source'] == 'K562 · ENCFFTEST'
    assert mirna['start'] == 546 and mirna['end'] == 553
    assert mirna['evidence'] == 'Predicted 8mer site' and mirna['source'] == 'TargetScan · vert_80'
    assert regulatory_sites(entry, evidence, 'another_program')['regulatory_sites'] == []
    assert evidence == before
    evidence['global'] = deepcopy(evidence['by_program']['p'])
    assert len(regulatory_sites(entry, evidence, 'p')['regulatory_sites']) == 2


@pytest.mark.parametrize('conflict', ['assembly', 'transcript_version', 'strand', 'cdna', 'invalid_coordinates', 'unverified'])
def test_conflicting_sites_are_not_projected(conflict):
    evidence = constraints();rbp, mirna = evidence['by_program']['p']
    if conflict == 'assembly':
        rbp['materialization_summary']['assembly'] = 'hg19'
    elif conflict == 'transcript_version':
        rbp['materialization_summary']['feature_scope']['reference_transcript'] = 'TX.2'
    elif conflict == 'strand':
        rbp['materialized_rows'][0][5] = '-'
    elif conflict == 'cdna':
        rbp['coordinate_space'] = 'cdna'
    elif conflict == 'invalid_coordinates':
        rbp['materialized_rows'][0][1] = 530
    else:
        mirna['materialization_summary']['site_records'][0]['sequence_projection']['status'] = 'unresolved'
    result = regulatory_sites(reference(), evidence, 'p')
    assert len(result['regulatory_sites']) == 1 and result['binding_annotation_notes']


def test_report_projection_uses_saved_materialized_evidence_without_modification(tmp_path):
    goal, _, _ = run_fixture(tmp_path)
    goal['isoform_transcripts'] = {'TX': reference()}
    goal['design_programs'][0]['transcript_id'] = 'TX'
    save(tmp_path, 'artifacts/intake/structured_goal.json', goal)
    save(tmp_path, 'region_constraint_evidence.json', {'materialized_constraints': constraints()})
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*.json')}
    data = project_run(tmp_path)
    context = data['contexts'][0]
    assert context['reference_features']['segments'][-1]['kind'] == 'utr3'
    assert len(context['regulatory_sites']) == 2
    markup = annotation_html(data['contexts'])
    assert '3′ UTR' in markup and 'K562' in markup and 'Predicted 8mer site' in markup
    assert '511–530' in markup and '546–553' in markup
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*.json')}


def test_overview_escapes_source_labels_and_keeps_exact_coordinates():
    entry = reference()
    context = {'id': '<context>', 'transcript': 'TX.1', 'reference_strand': '+',
               **project_exon_names(entry), 'reference_features': transcript_features(entry),
               **regulatory_sites(entry, constraints(), 'p')}
    context['regulatory_sites'][0].update(label='<script>alert(1)</script>', url='javascript:alert(1)')
    rendered = annotation_html([context])
    assert '<script>' not in rendered and 'href="javascript:' not in rendered
    assert '&lt;script&gt;' in rendered and 'chr1:511–530' in rendered
