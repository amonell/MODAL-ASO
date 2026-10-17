from copy import deepcopy

import run_goal_intake as runtime
from aso_agents.preservation_context import preservation_resolution_objectives, preservation_reference_context


def test_preserved_transcripts_are_resolved_without_inventing_abundance_direction(monkeypatch):
    changed, protected = 'ENST00000000001', 'ENST00000000002'
    goal = {'gene_symbol': 'TEST', 'species': 'human', 'intent': 'splice_modulation',
            'splice_isoform_objectives': [{'isoform_label': changed, 'transcript_id': changed,
                'objective_scope': 'specific_transcript', 'desired_abundance': 'decrease', 'role': 'pathogenic'}],
            'preservation_targets': [{'action': 'preserve', 'preservation_target_type': 'transcript',
                                     'transcript_id': protected, 'gene_symbol': 'TEST'}]}
    before = deepcopy(goal)
    def structure(tid):
        return {'seq_region_name': '8', 'strand': 1, 'versioned_id': tid + '.2',
                'exons_tx': [{'index': i, 'start': start, 'end': end, 'contains_coding': True}
                    for i, (start, end) in enumerate([(100, 150), (200, 230 if tid == changed else 260), (300, 350)], 1)]}
    monkeypatch.setattr(runtime, 'list_transcripts', lambda **_: {'transcripts': [{'id': changed}, {'id': protected}]})
    monkeypatch.setattr(runtime, '_fetch_transcript_structure', structure)
    monkeypatch.setattr(runtime, '_generate_splice_strategy_analysis', lambda *_, **__: None)
    monkeypatch.setattr(runtime, '_cached_authoritative_exon_nomenclature_mapping', lambda *_, **__: {'status': 'ordinal_default'})
    result = runtime._resolve_isoform_transcripts(goal, deepcopy(goal), allow_interactive=False)
    assert set(result['isoform_transcripts']) == {changed, protected}
    assert goal['splice_isoform_objectives'] == before['splice_isoform_objectives']
    assert goal['preservation_targets'] == before['preservation_targets']
    assert all('desired_abundance' not in x for x in preservation_resolution_objectives(goal))
    shared = next(s for s in result['splice_site_plan'] if s['site'] == 'acceptor' and s['position'] == 200)
    assert set(shared['present_in']) == {changed, protected}
    assert shared['uniqueness'] != 'unique'
    assert preservation_reference_context(goal, result['isoform_transcripts'])[0]['resolution_status'] == 'resolved'


def test_preservation_matching_does_not_alias_versions_or_other_genes():
    goal = {'gene_symbol': 'TEST', 'preservation_targets': [
        {'action': 'preserve', 'preservation_target_type': 'transcript', 'transcript_id': 'TX.3'},
        {'action': 'preserve', 'preservation_target_type': 'isoform', 'isoform_label': 'normal', 'gene_symbol': 'OTHER'}]}
    catalog = {'TX': {'transcript_id': 'TX', 'metadata': {'versioned_id': 'TX.2'}}}
    assert len(preservation_resolution_objectives(goal)) == 1
    assert all(x['resolution_status'] == 'unresolved' for x in preservation_reference_context(goal, catalog))
    assert runtime._suggest_safeguard_direction({'preserved_isoforms': ['TX'], 'decision_direction': 'increase'}) == ('neutral', 'preserve_requested_reference')
