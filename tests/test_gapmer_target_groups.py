"""Group specificity follows typed, verified features, never gene nicknames."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_goal_intake as runtime
from aso_agents import pipeline_tools
from aso_agents.curated_exon_nomenclature import reference_context, reference_fingerprint
from aso_agents.gapmer_target_groups import resolve_groups


def context(*, strand='+', presence='included', gene='SYNTHETIC_A'):
    def entry(identity, spans):
        if strand == '-':
            spans = list(reversed(spans))
        return {'transcript_id': identity, 'chrom': 'chr8', 'strand': strand,
                'metadata': {'id': identity, 'version': 4, 'assembly_name': 'GRCh38', 'species': 'homo_sapiens', 'gene_symbol': gene},
                'exons_tx': [{'index': i, 'start': start, 'end': end} for i, (start, end) in enumerate(spans, 1)]}
    catalog = {
        'included_a': entry('included_a', [(101, 150), (301, 350), (501, 550)]),
        'included_b': entry('included_b', [(51, 80), (101, 150), (301, 350), (501, 550)]),
        'excluded_a': entry('excluded_a', [(101, 150), (501, 550)]),
        'excluded_b': entry('excluded_b', [(51, 80), (101, 150), (501, 550)]),
        'partial': entry('partial', [(101, 150)]),
        'retained': entry('retained', [(101, 550)]),
    }
    reference = deepcopy(catalog['included_a'])
    ref = reference_context(reference)
    reference['metadata']['exon_nomenclature_resolutions'] = {'7b': {
        'status': 'verified_curated_mapping', 'requested_exon_label': '7b',
        'resolved_transcript_ordinal': 2, 'reference_transcript_id': ref['transcript_id'],
        'reference_fingerprint': reference_fingerprint(ref['transcript_id'], reference['exons_tx'], ref),
        'evidence_refs': ['synthetic-fixture:exon-names'],
    }}
    goal = {'gene_symbol': gene, 'species': 'human', 'intent': 'knockdown',
            'transcript_lock': {'kind': 'explicit_id', 'value': 'included_a.4'},
            'isoform_transcripts': {'numbering_reference': reference},
            'splice_isoform_objectives': [{
                'objective_scope': 'aggregate_class', 'isoform_label': 'requested group',
                'transcript_id': 'included_a.4', 'desired_abundance': 'decrease', 'role': 'context',
                'aggregate_class': {'basis': 'exon_presence', 'exon_label': '7b', 'presence': presence}}]}
    return goal, catalog


def annotate(tmp_path, goal, catalog, matches):
    rows = [{'Name': name, 'Sequence': sequence, 'GC': .5} for name, sequence, _ in matches]
    source = tmp_path / 'tiles.json'
    source.write_text(json.dumps({'rows': rows}))
    scope = runtime._resolve_gapmer_uniqueness_context(goal, 'numbering_reference')
    index = {len(seq): {} for _, seq, _ in matches}
    for _, seq, labels in matches:
        index[len(seq)][seq] = set(labels)
    output = runtime._annotate_gapmer_uniqueness(
        str(source), index, scope['target_label'], len(catalog),
        beneficial_isoform_labels=scope['beneficial_labels'], pathogenic_isoform_labels=scope['pathogenic_labels'],
        transcript_catalog=catalog, resolved_transcripts=goal['isoform_transcripts'], structured_goal=goal)
    assert json.loads(source.read_text())['rows'] == rows
    return Path(output), json.loads(Path(output).read_text())


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('presence', ['included', 'excluded'])
@pytest.mark.parametrize('gene', ['SYNTHETIC_A', 'UNRELATED_B'])
def test_membership_uses_verified_feature_and_observed_bypass(strand, presence, gene):
    goal, catalog = context(strand=strand, presence=presence, gene=gene)
    before = deepcopy((goal, catalog))
    receipt = resolve_groups(goal, catalog, goal['isoform_transcripts'])
    group = receipt['groups'][0]
    assert receipt['status'] == 'partial'
    assert group['members'] == [presence + '_a', presence + '_b']
    assert group['unknown'] == ['partial', 'retained']
    assert group['features'][0]['resolved_transcript_ordinal'] == 2
    assert group['features'][0]['requested_exon_label'] == '7b'
    assert (goal, catalog) == before


def test_reference_lock_does_not_turn_group_into_single_transcript():
    goal, _ = context()
    scope = runtime._resolve_gapmer_uniqueness_context(goal, 'numbering_reference')
    assert scope['mode'] == 'target_group_specific'
    assert scope['target_label'] is None and scope['target_labels'] == []
    assert scope['beneficial_labels'] == [] and scope['pathogenic_labels'] == []


def test_shared_group_site_outranks_reference_only_selectivity_bug(tmp_path):
    goal, catalog = context()
    path, data = annotate(tmp_path, goal, catalog, [
        ('shared_utr', 'AAAA', {'included_a', 'excluded_a', 'excluded_b'}),
        ('group_feature', 'CCCC', {'included_a', 'included_b', 'partial'})])
    rows = {r['Name']: r for r in data['rows']}
    assert rows['shared_utr']['isoform_uniqueness_score'] == 1/3
    feature = rows['group_feature']
    assert feature['target_isoform_matches'] == ['included_a', 'included_b']
    assert feature['isoform_uniqueness_score'] == 2/3
    assert feature['other_isoform_matches'] == []
    assert feature['unclassified_isoform_matches'] == ['partial']
    assert feature['target_group_coverage_fraction'] == 1
    assert feature['isoform_ranking_status'] == 'group_selectivity_lower_bound'
    ranking = runtime._annotate_gapmer_selection_ranks(str(path))
    ranked = json.loads(path.read_text())['rows']
    assert [r['Name'] for r in ranked] == ['group_feature', 'shared_utr']
    assert ranked[0]['selection_score_components']['target_isoform_catalog_size'] == 2
    assert any(c['criterion'] == 'target_group_coverage' for c in ranking['ordering'])
    from aso_agents.candidate_export_evidence import with_native_ranks, named_rank_components
    exported = with_native_ranks(ranked)
    assert exported[0]['native_rank'] == 1
    assert named_rank_components(exported[0])['target_group_coverage_fraction'] == 1


def test_equal_selectivity_rewards_target_group_coverage_instead_of_fewer_hits(tmp_path):
    goal, catalog = context()
    _, data = annotate(tmp_path, goal, catalog, [
        ('one_member', 'AAAA', {'included_a'}),
        ('all_members', 'CCCC', {'included_a', 'included_b'})])
    assert [r['Name'] for r in data['rows']] == ['all_members', 'one_member']
    assert [r['isoform_uniqueness_score'] for r in data['rows']] == [1, 1]
    assert [r['target_group_coverage_fraction'] for r in data['rows']] == [1, .5]


@pytest.mark.parametrize('fault', ['no_mapping', 'wrong_version', 'stale_coordinates', 'unsupported_class', 'empty_catalog'])
def test_unknown_group_never_falls_back_to_the_numbering_transcript(tmp_path, fault):
    goal, catalog = context()
    reference = goal['isoform_transcripts']['numbering_reference']
    if fault == 'no_mapping': reference['metadata'].pop('exon_nomenclature_resolutions')
    if fault == 'wrong_version': goal['splice_isoform_objectives'][0]['transcript_id'] = 'included_a.5'
    if fault == 'stale_coordinates': reference['exons_tx'][1]['end'] += 1
    if fault == 'unsupported_class': goal['splice_isoform_objectives'][0]['aggregate_class'] = {'basis': 'role', 'role': 'pathogenic'}
    if fault == 'empty_catalog': catalog = {}
    _, data = annotate(tmp_path, goal, catalog, [('site', 'AAAA', {'included_a'})])
    row = data['rows'][0]
    assert data['isoform_group_resolution']['status'] == 'unresolved'
    assert row['isoform_uniqueness_score'] is None
    assert row['target_isoform_match_count'] is None
    assert row['isoform_uniqueness_class'] == 'unresolved_role_identity'
    assert row['target_group_coverage_fraction'] is None
    assert row['isoform_ranking_status'] == 'unresolved_group_sequence_and_risk_only'


@pytest.mark.parametrize('fault', ['assembly', 'species', 'strand', 'chromosome', 'overlap', 'missing_structure', 'partial_overlap'])
def test_incompatible_or_partial_structures_are_unknown(fault):
    goal, catalog = context()
    entry = catalog['included_b']
    if fault == 'assembly': entry['metadata']['assembly_name'] = 'GRCh37'
    if fault == 'species': entry['metadata']['species'] = 'mus_musculus'
    if fault == 'strand': entry['strand'] = '-'
    if fault == 'chromosome': entry['chrom'] = 'chr9'
    if fault == 'overlap': entry['exons_tx'][1]['end'] = 330
    if fault == 'missing_structure': entry['exons_tx'] = []
    if fault == 'partial_overlap': entry['exons_tx'][2]['end'] = 325
    group = resolve_groups(goal, catalog, goal['isoform_transcripts'])['groups'][0]
    assert 'included_b' in group['unknown']
    assert 'included_b' not in group['members'] + group['nonmembers']


def test_external_reference_alias_requires_attestation():
    goal, catalog = context()
    obj = goal['splice_isoform_objectives'][0]
    obj.pop('transcript_id'); obj['reference_accession'] = 'NM_123456.9'
    assert resolve_groups(goal, catalog, goal['isoform_transcripts'])['status'] == 'unresolved'
    reference = goal['isoform_transcripts']['numbering_reference']
    reference['metadata']['reference_accession_resolution'] = {
        'requested_accession': 'NM_123456.9', 'exact_accession_verified': True}
    assert resolve_groups(goal, catalog, goal['isoform_transcripts'])['groups'][0]['members'] == ['included_a', 'included_b']
    obj['reference_accession'] = 'NM_123456.8'
    assert resolve_groups(goal, catalog, goal['isoform_transcripts'])['status'] == 'unresolved'


def test_beneficial_group_is_separate_and_unknowns_have_conservative_penalty(tmp_path):
    goal, catalog = context()
    protected = deepcopy(goal['splice_isoform_objectives'][0])
    protected.update(isoform_label='preserved group', desired_abundance='increase')
    protected['aggregate_class']['presence'] = 'excluded'
    goal['splice_isoform_objectives'].append(protected)
    _, data = annotate(tmp_path, goal, catalog, [('site', 'AAAA', {'included_a', 'excluded_a', 'partial'})])
    row = data['rows'][0]
    assert row['target_isoform_matches'] == ['included_a']
    assert row['beneficial_isoform_matches'] == ['excluded_a']
    assert row['unclassified_isoform_matches'] == ['partial']
    assert row['isoform_uniqueness_score'] == 1/9
    assert row['isoform_uniqueness_class'] == 'target_shared_with_beneficial'


def test_conflicting_target_and_protected_group_is_unresolved(tmp_path):
    goal, catalog = context()
    protected = deepcopy(goal['splice_isoform_objectives'][0])
    protected['desired_abundance'] = 'increase'
    goal['splice_isoform_objectives'].append(protected)
    _, data = annotate(tmp_path, goal, catalog, [('site', 'AAAA', {'included_a'})])
    assert data['rows'][0]['isoform_uniqueness_score'] is None
    assert data['rows'][0]['isoform_role_resolution']['conflicting_target_and_beneficial_labels'] == ['included_a', 'included_b']


def test_sequence_backed_catalog_defines_coverage_denominator(tmp_path):
    goal, catalog = context()
    catalog.pop('included_b')
    _, data = annotate(tmp_path, goal, catalog, [('site', 'AAAA', {'included_a'})])
    assert data['rows'][0]['target_isoform_catalog_size'] == 1
    assert data['rows'][0]['target_group_coverage_fraction'] == 1


def test_explicit_single_transcript_outcome_can_supplement_a_group(tmp_path):
    goal, catalog = context()
    goal['splice_isoform_objectives'].append({'objective_scope': 'specific_transcript',
        'transcript_id': 'excluded_a.4', 'desired_abundance': 'decrease'})
    _, data = annotate(tmp_path, goal, catalog, [('site', 'AAAA', {'included_a', 'excluded_a'})])
    row = data['rows'][0]
    assert row['target_isoform_matches'] == ['excluded_a', 'included_a']
    assert row['target_isoform_catalog_size'] == 3


def test_report_exports_group_coverage_separately_from_whole_catalog(tmp_path):
    from aso_agents.report_explorer_data import project_records
    from aso_agents.html_report import _build_gapmer_explanation
    from aso_agents.report_attributes import COMMON
    goal, catalog = context()
    _, data = annotate(tmp_path, goal, catalog, [('site', 'AAAA', {'included_a', 'included_b'})])
    source = {**data['rows'][0], 'antisense_5to3': 'TTTT', 'ordering_notation': 'TTTT'}
    projected = project_records([source])
    attrs = {a['label']: a for a in projected['attributes']}
    assert attrs['Target group coverage']['values'] == [1]
    assert attrs['Isoform coverage']['values'] == [1/3]
    assert 'Original candidate order' in next(a[2] for a in COMMON if a[0] == 'native_rank')
    explanation = _build_gapmer_explanation(tmp_path, {}, {**goal, 'gapmer_tiling_results': {'row_count': 1}}, {})
    assert 'requested transcript group' in json.dumps(explanation)
    assert 'isoform_group_definitions' in pipeline_tools._PROVIDED_ASO_SCORED_TILE_FIELDS


def test_direct_caller_cannot_substitute_reference_for_excluded_group(tmp_path):
    goal, catalog = context(presence='excluded')
    source = tmp_path / 'tiles.json'
    source.write_text(json.dumps({'rows': [{'Sequence': 'AAAA'}]}))
    path = runtime._annotate_gapmer_uniqueness(str(source), {4: {'AAAA': {'included_a', 'excluded_a'}}},
        'included_a.4', len(catalog), transcript_catalog=catalog,
        resolved_transcripts=goal['isoform_transcripts'], structured_goal=goal)
    row = json.loads(Path(path).read_text())['rows'][0]
    assert row['target_isoform_matches'] == ['excluded_a']
    assert row['other_isoform_matches'] == ['included_a']


def test_multiple_requested_groups_are_a_union_without_protecting_unrequested_classes(tmp_path):
    goal, catalog = context()
    second = deepcopy(goal['splice_isoform_objectives'][0])
    second['aggregate_class']['presence'] = 'excluded'
    goal['splice_isoform_objectives'].append(second)
    _, data = annotate(tmp_path, goal, catalog, [('site', 'AAAA', {'included_a', 'excluded_a'})])
    row = data['rows'][0]
    assert row['target_isoform_matches'] == ['excluded_a', 'included_a']
    assert row['beneficial_isoform_matches'] == []
    assert row['target_isoform_catalog_size'] == 4
    assert row['isoform_uniqueness_score'] == 1


@pytest.mark.parametrize('grouped', [True, False])
def test_missing_sequence_index_is_not_a_measured_zero(tmp_path, grouped):
    goal, catalog = context()
    source = tmp_path / 'tiles.json'
    source.write_text(json.dumps({'rows': [{'Sequence': 'AAAAA'}]}))
    path = runtime._annotate_gapmer_uniqueness(str(source), {4: {'AAAA': {'included_a'}}},
        None, len(catalog), transcript_catalog=catalog,
        resolved_transcripts=goal['isoform_transcripts'], structured_goal=goal if grouped else None)
    row = json.loads(Path(path).read_text())['rows'][0]
    assert row['isoform_uniqueness_score'] is None
    assert row['isoform_hit_count'] is None


def test_explicit_ordinal_group_survives_normalization_without_claiming_a_public_name():
    goal, catalog = context()
    goal['isoform_transcripts']['numbering_reference']['metadata'].pop('exon_nomenclature_resolutions')
    definition = {'basis': 'exon_presence', 'exon_index': 2, 'presence': 'included'}
    goal['splice_isoform_objectives'][0]['aggregate_class'] = runtime._normalize_splice_aggregate_class(definition)
    group = resolve_groups(goal, catalog, goal['isoform_transcripts'])['groups'][0]
    assert group['members'] == ['included_a', 'included_b']
    assert group['features'][0]['status'] == 'explicit_transcript_ordinal'
    assert not group['features'][0]['biological_label_verified']


def test_unresolved_new_scoring_clears_old_selectivity_in_supplied_aso_evidence(tmp_path):
    goal, catalog = context()
    goal['isoform_transcripts']['numbering_reference']['metadata'].pop('exon_nomenclature_resolutions')
    _, data = annotate(tmp_path, goal, catalog, [('site', 'AAAA', {'included_a'})])
    scored = {**data['rows'][0], 'provided_row_index': 0}
    rows = [{'sequence': 'TTTT', 'isoform_uniqueness_score': 1, 'target_isoform_matches': ['old_reference']}]
    pipeline_tools._merge_scored_tiles_into_provided_asos(rows, [scored])
    assert rows[0]['isoform_uniqueness_score'] is None
    assert rows[0]['target_isoform_matches'] is None
    assert rows[0]['stage_results']['uniqueness_scoring']['status'] == 'unavailable'
    assert rows[0]['isoform_group_definitions'] == scored['isoform_group_definitions']


def test_classification_is_once_per_catalog_not_once_per_candidate(tmp_path, monkeypatch):
    import aso_agents.gapmer_target_groups as groups
    goal, catalog = context()
    original = groups.exon_membership
    calls = []
    def classify(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(groups, 'exon_membership', classify)
    annotate(tmp_path, goal, catalog, [(str(i), 'AAAA', {'included_a'}) for i in range(1000)])
    assert len(calls) == len(catalog)


def test_pipeline_scoring_carries_group_and_does_not_restore_reference_fallback(tmp_path, monkeypatch):
    goal, catalog = context()
    source = tmp_path / 'tiles.json'
    source.write_text(json.dumps({'rows': [{'Name': 'group_site', 'Sequence': 'AAAA'}]}))
    monkeypatch.setattr(pipeline_tools, '_annotate_gapmer_offtargets_by_species', lambda **kw: {})
    monkeypatch.setattr(pipeline_tools, '_annotate_repeat_risk_json', lambda **kw: {})
    fake = SimpleNamespace(
        _apply_initial_tiling_filters=lambda path, constraints: {'passed': {'path': path}},
        _resolve_patient_reference_fasta=lambda goal: None,
        _build_isoform_cdna_sequences=lambda *a, **kw: {label: 'AAAA' for label in catalog},
        _build_gapmer_kmer_index_by_len=lambda *a: {4: {'AAAA': {'included_a', 'included_b'}}},
        _resolve_gapmer_uniqueness_context=runtime._resolve_gapmer_uniqueness_context,
        _resolve_uniqueness_target_label=lambda *a: pytest.fail('Group cannot fall back to reference'),
        _annotate_gapmer_uniqueness=runtime._annotate_gapmer_uniqueness,
    )
    result = pipeline_tools._score_gapmer_tiles_json(
        gapmer_results={'json_path': str(source)}, structured_goal=goal, runtime=fake,
        isoform_transcripts=goal['isoform_transcripts'], all_transcripts_full=catalog,
        iso_label_primary='numbering_reference')
    assert 'error' not in result, result
    row = json.loads(Path(result['scored_tiles_path']).read_text())['rows'][0]
    assert row['target_isoform_matches'] == ['included_a', 'included_b']
    assert row['isoform_uniqueness_score'] == 1
