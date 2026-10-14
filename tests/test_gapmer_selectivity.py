import json
from copy import deepcopy

import pytest

from aso_chemistry import encode_idt_oligo
from aso_agents.gapmer_selectivity import (
    annotate_preservation, bind_uniqueness_context, explicit_junction_region,
    filter_preservation, resolve_catalog_label,
)


def transcript(identity, intervals, strand='+'):
    return {'transcript_id': identity, 'chrom': 'chr8', 'strand': strand,
            'exons_tx': [{'index': i, 'start': a, 'end': b}
                         for i, (a, b) in enumerate(intervals, 1)]}


def references(strand='+'):
    coords = [(101, 180), (301, 330), (501, 600), (701, 760)]
    if strand == '-':
        coords.reverse()
    return {'short': transcript('TX1', [coords[0], coords[2], coords[3]], strand),
            'long': transcript('TX2', coords, strand)}


def protected_goal():
    return {'preservation_targets': [{'action': 'preserve',
             'preservation_target_type': 'isoform', 'isoform_label': 'long'}]}


@pytest.mark.parametrize('strand', ['+', '-'])
def test_exon_pair_maps_through_attested_protected_structure_and_requires_crossing(strand):
    refs = references(strand)
    program = {'target_type': 'splice_junction', 'isoform_label': 'short',
               'transcript_id': 'TX1', 'exon_labels': [1, 3]}
    result = explicit_junction_region(protected_goal(), program, refs, 'short')
    first = refs['short']['exons_tx'][0]
    boundary = first['end'] - first['start'] + 1
    assert result['junction_boundary_cdna'] == boundary
    assert result['rows'][0][1:3] == [boundary - 1, boundary + 1]
    assert result['tile_overlap_requirement'] == 'contains_feature'
    assert result['junction_mapping'][0]['target_exon_ordinals'] == [1, 2]
    assert result['junction_mapping'][0]['requested_exon_labels'] == [1, 3]


def test_direct_adjacent_pair_and_verified_nomenclature():
    refs = references()
    program = {'target_type': 'splice_junction', 'isoform_label': 'short', 'exon_labels': [1, 2]}
    assert explicit_junction_region({}, program, refs, 'short')['junction_boundary_cdna'] == 80
    refs['short']['exon_nomenclature_resolutions'] = {'3': {
        'status': 'verified_alternate_mapping', 'resolved_transcript_ordinal': 2}}
    program['exon_labels'] = [1, 3]
    assert explicit_junction_region({}, program, refs, 'short')['junction_boundary_cdna'] == 80


def test_missing_conflicting_and_ambiguous_endpoints_do_not_broaden_to_neighboring_sites():
    refs = references()
    program = {'target_type': 'splice_junction', 'isoform_label': 'short', 'exon_labels': [1, 3]}
    assert 'error' in explicit_junction_region({}, program, refs, 'short')
    program['transcript_id'] = 'TX2'
    assert 'error' in explicit_junction_region(protected_goal(), program, refs, 'short')
    program.pop('transcript_id'); program['exon_labels'] = [3, 1]
    assert 'error' in explicit_junction_region(protected_goal(), program, refs, 'short')
    assert explicit_junction_region({}, {'target_type': 'gene'}, refs, 'short') is None


def test_alias_binding_preserves_catalog_population_and_version_conflicts():
    refs = references(); catalog = {entry['transcript_id']: deepcopy(entry) for entry in refs.values()}
    context = {'target_label': 'short', 'target_labels': ['short'],
               'beneficial_labels': ['long'], 'pathogenic_labels': ['short']}
    bound = bind_uniqueness_context(context, catalog, refs)
    assert bound['target_label'] == 'TX1'
    assert bound['beneficial_labels'] == ['TX2']
    assert len(catalog) == 2 and context['target_label'] == 'short'
    catalog['TX1']['metadata'] = {'versioned_id': 'TX1.2'}
    refs['short']['metadata'] = {'versioned_id': 'TX1.1'}
    assert resolve_catalog_label('short', catalog, refs) is None
    assert resolve_catalog_label('TX1.9', catalog, refs) is None
    with pytest.raises(ValueError, match='identity_unresolved'):
        bind_uniqueness_context(context, catalog, refs)


def row(target, chemistry):
    antisense = target.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    return {'Sequence': target, 'Synthesis': antisense,
            'IDT': encode_idt_oligo(antisense, chemistry)}


@pytest.mark.parametrize('chemistry,target,gap_position', [
    ('2-5-3', 'ACGTACGTGA', 4), ('3-6-2', 'TCGATAGCTGA', 5),
    ('2ome:2-5-3', 'ACGTACGTGA', 4)])
def test_actual_gap_positions_and_full_protected_sequence_search(tmp_path, chemistry, target, gap_position):
    def mutated(aso_position):
        pos = len(target) - aso_position
        return target[:pos] + ('C' if target[pos] != 'C' else 'A') + target[pos + 1:]
    path = tmp_path/'tiles.json'; original = row(target, chemistry)
    for protected, expected, positions in [(target, False, []), (mutated(1), False, [1]),
                                          (mutated(gap_position), True, [gap_position])]:
        path.write_text(json.dumps({'rows': [original]}))
        result = annotate_preservation(path, {'preservation_targets':[{'action':'preserve','preservation_target_type':'isoform','isoform_label':'protected'}]}, {'beneficial_labels': ['protected']},
                                       {'protected': protected}, chemistry)
        assert result['status'] == 'complete'
        data = json.loads(path.read_text())['rows'][0]['gapmer_preservation_assessment']
        assert data['recommended_for_selective_panel'] is expected
        assert data['protected_references'][0]['closest_sites'][0]['mismatch_positions_aso_1based'] == positions
        filtered = filter_preservation(path)
        assert filtered['passing_count'] == int(expected)
        assert filtered['failed_count'] == int(not expected)
        assert json.loads(path.read_text())['rows'][0]['Synthesis'] == original['Synthesis']


def test_every_equally_close_site_and_missing_reference_is_not_zero(tmp_path):
    target = 'ACGTACGTGA'; path = tmp_path/'tiles.json'
    path.write_text(json.dumps({'rows': [row(target, '2-5-3')]}))
    annotate_preservation(path, {'preservation_targets':[{'action':'preserve','preservation_target_type':'isoform','isoform_label':'protected'}]}, {'beneficial_labels': ['protected']},
                          {'protected': target + 'GGGG' + target}, '2-5-3')
    data = json.loads(path.read_text())['rows'][0]['gapmer_preservation_assessment']
    assert [x['transcript_cdna_start_1based'] for x in data['protected_references'][0]['closest_sites']] == [1, 15]
    with pytest.raises(ValueError, match='sequence_incomplete'):
        annotate_preservation(path, {'preservation_targets':[{'action':'preserve','preservation_target_type':'isoform','isoform_label':'protected'}]}, {'beneficial_labels': ['protected']}, {}, '2-5-3')
    with pytest.raises(ValueError, match='physical_chemistry_unverified'):
        annotate_preservation(path, {'preservation_targets':[{'action':'preserve','preservation_target_type':'isoform','isoform_label':'protected'}]}, {'beneficial_labels': ['protected']}, {'protected': target}, '3-5-2')
    assert annotate_preservation(path, {}, {}, {}, None)['status'] == 'not_required'


def test_empty_panel_retains_schema_and_full_rejection_evidence(tmp_path):
    target = 'ACGTACGTGA'; path = tmp_path/'tiles.json'
    path.write_text(json.dumps({'schema': {'columns': {'Sequence': 'string'}}, 'rows': [row(target, '2-5-3')]}))
    annotate_preservation(path, {'preservation_targets':[{'action':'preserve','preservation_target_type':'isoform','isoform_label':'protected'}]}, {'beneficial_labels': ['protected']}, {'protected': target}, '2-5-3')
    result = filter_preservation(path)
    from pathlib import Path
    assert json.loads(Path(result['output_path']).read_text())['rows'] == []
    assert len(json.loads(Path(result['excluded_path']).read_text())['rows']) == 1
    assert 'gapmer_preservation_assessment' in json.loads(Path(result['output_path']).read_text())['schema']['columns']
