from copy import deepcopy
import json
from pathlib import Path

import pytest
import run_goal_intake as runtime
from aso_agents.gapmer_selectivity import resolve_catalog_label


def entry(identity='TX17', version=4):
    return {'transcript_id': identity, 'chrom': 'chr2', 'strand': '+',
            'exons_tx': [{'index': 1, 'start': 101, 'end': 200}],
            'metadata': {'id': identity, 'version': version,
                         'assembly_name': 'GRCh38', 'species': 'homo_sapiens'}}


def test_version_is_attested_by_reference_record_and_does_not_change_catalog_keys():
    catalog = {'catalog_row': entry()}
    before = deepcopy(catalog)
    assert resolve_catalog_label('TX17.4', catalog, {}) == 'catalog_row'
    assert resolve_catalog_label('TX17', catalog, {}) == 'catalog_row'
    assert resolve_catalog_label('TX17.5', catalog, {}) is None
    assert catalog == before


@pytest.mark.parametrize('fault', ['duplicate', 'wrong_version', 'conflicting_metadata',
                                  'conflicting_key', 'source_version', 'assembly', 'species', 'structure'])
def test_ambiguity_and_conflicting_reference_evidence_remain_unresolved(fault):
    catalog = {'TX17': entry()}
    resolved = {'requested': entry()}
    query = 'TX17.4'
    if fault == 'duplicate':
        catalog['duplicate'] = entry()
    elif fault == 'wrong_version':
        catalog['TX17']['metadata']['version'] = 5
    elif fault == 'conflicting_metadata':
        catalog['TX17']['metadata']['versioned_id'] = 'TX17.5'
    elif fault == 'conflicting_key':
        catalog = {'TX17.5': entry()}
    elif fault == 'source_version':
        query = 'TX17'
        resolved['requested']['metadata']['version'] = 5
    elif fault in {'assembly', 'species'}:
        field = 'assembly_name' if fault == 'assembly' else 'species'
        resolved['requested']['metadata'][field] = 'GRCm39' if fault == 'assembly' else 'mus_musculus'
    else:
        resolved['requested']['exons_tx'][0]['end'] = 201
    assert resolve_catalog_label(query, catalog, resolved) is None


def test_named_alias_joins_only_complete_same_reference_structure_and_attested_version():
    catalog = {'TX17': entry()}
    resolved = {'short': entry()}
    assert resolve_catalog_label('short', catalog, resolved) == 'TX17'
    for key in ('chrom', 'exons_tx'):
        incomplete = deepcopy(catalog)
        incomplete['TX17'].pop(key)
        assert resolve_catalog_label('short', incomplete, resolved) is None
    catalog['TX17']['metadata'].pop('version')
    assert resolve_catalog_label('short', catalog, resolved) is None


def annotate(tmp_path, *, target='TX17.4', beneficial=None, catalog=None, resolved=None, hits=None):
    catalog = catalog if catalog is not None else {'TX17': entry(), 'TX29': entry('TX29', 2), 'other': entry('TX81', 1)}
    rows = [{'Name': 'candidate', 'Sequence': 'ACGTACTG', 'Synthesis': 'CAGTACGT'}]
    source = tmp_path/'tiles.json'
    source.write_text(json.dumps({'rows': rows}))
    output = runtime._annotate_gapmer_uniqueness(
        str(source), {8: {'ACGTACTG': set(catalog) if hits is None else hits}}, target, len(catalog),
        beneficial_isoform_labels=beneficial, pathogenic_isoform_labels=[target] if target else [],
        transcript_catalog=catalog, resolved_transcripts=resolved or {},
    )
    assert json.loads(source.read_text())['rows'] == rows
    return json.loads(Path(output).read_text())['rows'][0]


def test_target_other_and_protected_roles_use_catalog_identity_without_changing_sequence_hits(tmp_path):
    row = annotate(tmp_path, beneficial=['TX29.2'])
    assert row['isoform_uniqueness_labels'] == ['TX17', 'TX29', 'other']
    assert row['target_isoform_matches'] == ['TX17']
    assert row['pathogenic_isoform_matches'] == ['TX17']
    assert row['beneficial_isoform_matches'] == ['TX29']
    assert row['other_isoform_matches'] == ['other']
    assert row['isoform_uniqueness_score'] == 1/6
    assert row['isoform_uniqueness_class'] == 'target_shared_with_beneficial'
    assert row['isoform_role_resolution']['target']['bindings'] == [
        {'requested_label': 'TX17.4', 'catalog_label': 'TX17', 'status': 'resolved'}]


def test_auxiliary_unknown_identity_is_not_a_measured_zero_or_non_target_class(tmp_path):
    row = annotate(tmp_path, target='TX17.99')
    assert row['isoform_hit_count'] == 3
    assert row['isoform_coverage_fraction'] == 1
    assert row['target_isoform_match_count'] is None
    assert row['target_isoform_matches'] is None
    assert row['other_isoform_matches'] is None
    assert row['isoform_uniqueness_score'] is None
    assert row['isoform_uniqueness_class'] == 'unresolved_role_identity'
    assert row['isoform_role_resolution']['status'] == 'unresolved'


def test_unresolved_protected_role_does_not_erase_known_target_hit_or_invent_selectivity(tmp_path):
    row = annotate(tmp_path, beneficial=['TX29.99'])
    assert row['target_isoform_match_count'] == 1
    assert row['beneficial_isoform_matches'] is None
    assert row['other_isoform_matches'] is None
    assert row['isoform_uniqueness_score'] is None


def test_known_reference_without_a_sequence_hit_remains_measured_zero(tmp_path):
    row = annotate(tmp_path, hits={'other'})
    assert row['target_isoform_match_count'] == 0
    assert row['target_isoform_matches'] == []
    assert row['other_isoform_matches'] == ['other']
    assert row['isoform_uniqueness_score'] == 0
    assert row['isoform_uniqueness_class'] == 'non_target'


def test_gene_wide_scope_retains_coverage_fraction(tmp_path):
    row = annotate(tmp_path, target=None, hits={'TX17', 'other'})
    assert row['isoform_uniqueness_score'] == 2/3
    assert row['isoform_uniqueness_class'] == 'gene_wide'
    assert row['target_isoform_matches'] == []


def test_missing_sequence_catalog_does_not_validate_versioned_target_by_stripping(tmp_path):
    row = annotate(tmp_path, catalog={'TX17': {}}, hits={'TX17'})
    assert row['isoform_hit_count'] == 1
    assert row['target_isoform_match_count'] is None
    assert row['isoform_uniqueness_score'] is None
