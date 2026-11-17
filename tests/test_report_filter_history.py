"""Filter history must expose exclusions without changing selection semantics."""
from collections import Counter
from html.parser import HTMLParser
import json

import pytest

from aso_agents.report_filter_history import project_filter_history
from aso_agents.report_filter_history_ui import filter_history_html
from aso_agents.readable_report import basic_document
from aso_agents.html_report import render_goal_intake_html


def save(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def native(start=100, **kwargs):
    return {'Synthesis': 'AAAAACGT', 'Sequence': 'ACGTTTTT', 'IDT': 'A*A*A*A*A*C*G*T',
            'Chrom': 'chr1', 'Start': start, 'End': start+8, 'Strand': '+', 'GC': .25,
            'Homopolymer_excluded_motifs': '["AAAAA","TTTTT"]',
            'Homopolymer_sequence_role': 'synthesized_antisense_sequence', **kwargs}


def trace(root, rows, raw):
    path = save(root, 'artifacts/design/tiling/tiles.json', {'rows': raw})
    return save(root, 'artifacts/candidates/generated_filter_trace.json',
                {'rows': rows, 'sources': {'raw_generated': [str(path)]}})


def drop(sequence='ACGTTTTT', terminal='filtered_during_scoring_or_ranking'):
    return {'sequence': sequence, 'terminal_classification': terminal, 'eligible': False,
            'stage_results': {'ranking': {'status': 'fail'}}}


class Packet(HTMLParser):
    def __init__(self, document):
        super().__init__(); self.active = False; self.value = ''; self.tr = 0
        self.feed(document)
    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('class') == 'filter-history-data':
            self.active = True
        if tag == 'tr':
            self.tr += 1
    def handle_data(self, data):
        if self.active:
            self.value += data
    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False


@pytest.mark.parametrize('folder', ['tiling', 'gapmer_tiling', 'junction_gapmer_tiling', 'steric_jobs/j', 'gapmer_jobs/j'])
def test_native_exclusions_keep_physical_sequence_coordinates_and_evidence(tmp_path, folder):
    path = save(tmp_path, f'artifacts/design/{folder}/filtered_no_homopolymer.json', {'rows': [native(Strand='-')]})
    before = path.read_bytes()
    data = project_filter_history(tmp_path)
    row, = data['records']
    assert row['aso'] == 'AAAAACGT' and row['strand'] == '-'
    assert row['intervals'] == [{'chrom': 'chr1', 'start': 101, 'end': 108}]
    assert row['reason'] == 'Excluded homopolymer in synthesized ASO: AAAAA'
    assert row['sources'] == [str(path.relative_to(tmp_path))]
    assert row['gc'] == .25 and row['notation'] == native()['IDT']
    assert path.read_bytes() == before


def test_aggregate_copies_merge_but_jobs_chemistries_and_bindings_do_not(tmp_path):
    row = native()
    save(tmp_path, 'artifacts/design/tiling/filtered_no_homopolymer.json', {'rows': [row]})
    for job in ('a', 'b'):
        save(tmp_path, f'artifacts/design/steric_jobs/{job}/filtered_no_homopolymer.json', {'rows': [row]})
    save(tmp_path, 'artifacts/design/gapmer_jobs/c/filtered_no_homopolymer.json', {'rows': [
        native(IDT='AAAAACGT'), native(200)]})
    data = project_filter_history(tmp_path)
    assert len(data['records']) == 4
    assert Counter(r['job'] for r in data['records']) == {'a': 1, 'b': 1, 'c': 2}
    assert len(next(r for r in data['records'] if r['job'] == 'a')['sources']) == 2


def test_custom_receipt_and_cross_species_target_rna_buckets(tmp_path):
    custom = save(tmp_path, 'artifacts/design/custom-rejections.json', {'rows': [native()]})
    goal = {'some_new_modality': {'new_admission_check': {'excluded_path': str(custom),
             'stage': 'Requested region', 'reason': 'No overlap with the requested region.'}}}
    save(tmp_path, 'artifacts/intake/structured_goal.json', goal)
    save(tmp_path, 'artifacts/design/gapmer_tiling/initial_filtering_tiles.cross_species_excluded.json', {'rows': [native(200)]})
    save(tmp_path, 'artifacts/design/tiling/initial_filtering_tiles_outside_target_rna.json', {'rows': [native(300)]})
    data = project_filter_history(tmp_path)
    assert len(data['records']) == 3
    assert {r['stage'] for r in data['records']} == {'Requested region', 'Required species matching', 'Target-RNA admission'}
    assert any(r['reason'] == 'No overlap with the requested region.' for r in data['records'])


def test_many_jobs_read_shared_admission_artifact_once(tmp_path, monkeypatch):
    from aso_agents import report_filter_history as module
    path = save(tmp_path, 'artifacts/design/tiling/filtered_no_homopolymer.json', {'rows':[native()]})
    jobs = [f'job_{i}' for i in range(40)]
    save(tmp_path, 'artifacts/intake/structured_goal.json', {
        'steric_design_jobs':[{'job_id':job} for job in jobs],
        'results':{job:{'excluded_path':str(path)} for job in jobs}})
    reads = Counter(); original = module.read_record
    def counted(source):
        reads[source] += 1
        return original(source)
    monkeypatch.setattr(module, 'read_record', counted)
    data = project_filter_history(tmp_path)
    assert reads[path] == 1
    assert len(data['records']) == 40
    assert {row['job'] for row in data['records']} == set(jobs)


def test_filtered_survivor_files_and_unselected_candidates_are_not_rejections(tmp_path):
    for name in ('filtered_tiles.json', 'scored_tiles.filtered.json', 'initial_filtering_tiles.json'):
        save(tmp_path, 'artifacts/design/tiling/'+name, {'rows': [native()]})
    trace(tmp_path, [drop(terminal='not_selected_final_output')], [native()])
    save(tmp_path, 'artifacts/candidates/candidate_assessments.json', {'rows': [
        {**native(), 'candidate_assessment': {'selection_eligible': True, 'recommendation_status': 'exploratory',
                                             'splice_safeguards': {'status': 'failed'}}}]})
    assert project_filter_history(tmp_path) is None


def test_unknown_trace_reason_is_not_invented_and_known_rejections_are_not_duplicated(tmp_path):
    a, b = native(), native(200, Synthesis='CGTCGTCG', Sequence='CGACGACG')
    save(tmp_path, 'artifacts/design/tiling/filtered_no_homopolymer.json', {'rows': [a]})
    trace(tmp_path, [drop(terminal='filtered_during_initial_filter'), drop('CGACGACG')], [a,b])
    data = project_filter_history(tmp_path)
    assert len(data['records']) == 2
    unknown, = [r for r in data['records'] if r['status'] == 'Removal reason unavailable']
    assert unknown['aso'] == 'CGTCGTCG'
    assert 'No candidate-specific exclusion reason was recorded' in unknown['reason']
    assert unknown['stage'] == 'Before or during scoring'


@pytest.mark.parametrize('difference', ['position', 'chemistry', 'job'])
def test_sequence_only_trace_never_guesses_an_ambiguous_physical_candidate(tmp_path, difference):
    rows = [native()]
    if difference == 'position': rows.append(native(200))
    if difference == 'chemistry': rows.append(native(IDT='AAAAACGT'))
    if difference == 'job':
        for job in ['a', 'b']:
            save(tmp_path, f'artifacts/design/steric_jobs/{job}/tiles.json', {'rows': [native()]})
    trace(tmp_path, [drop()], rows)
    result, = project_filter_history(tmp_path)['records']
    assert result['aso'] == '' and result['intervals'] == []
    assert result['comparison_sequence'] == 'ACGTTTTT'
    assert 'does not identify a unique ASO' in result['reason']


def test_unknown_orientation_and_coordinate_convention_stay_unknown(tmp_path):
    save(tmp_path, 'artifacts/candidates/candidate_exclusions.json', {'rows': [
        {'sequence': 'ACGT', 'chrom': 'chr2', 'start': 0, 'end': 4, 'reason': 'Recorded exclusion'},
        {'sequence': 'GGGG', 'chrom': 'chr2', 'start': 0, 'end': 4, 'reason': 'Another exclusion'}]})
    rows = project_filter_history(tmp_path)['records']
    assert len(rows) == 2
    assert all(not r['aso'] and not r['intervals'] and r['gc'] is None for r in rows)
    assert {r['reason'] for r in rows} == {'Recorded exclusion', 'Another exclusion'}


def test_supplied_candidates_keep_declared_roles_and_junction_segments(tmp_path):
    save(tmp_path, 'artifacts/design/v3_tasks/eval/provided_aso_evaluation.json', {'excluded_rows': [
        {'sequence': 'ACGT', 'declared_sequence_role': 'antisense_oligo_sequence', 'ordering_notation': 'ACGT',
         'genomic_segments': [{'chrom':'chr2', 'start':101, 'end':102}, {'chrom':'chr2', 'start':201, 'end':202}],
         'stage': 'Input validation', 'reason': 'Required candidate metadata missing.'}]})
    row, = project_filter_history(tmp_path)['records']
    assert row['aso'] == 'ACGT' and len(row['intervals']) == 2
    assert row['intervals'][0]['start'] == 101
    assert row['stage'] == 'Input validation'
    assert 'Required candidate metadata missing.' == row['reason']


def test_selection_eligibility_exclusions_are_separate_from_exploratory_predictions(tmp_path):
    save(tmp_path, 'artifacts/candidates/candidate_assessments.json', {'rows': [
        {**native(), 'candidate_assessment': {'selection_eligible': False,
            'motifs': {'flags': [{'action': 'exclude', 'reason': 'Requested motif rule failed.'}]}}},
        {**native(200), 'candidate_assessment': {'selection_eligible': True, 'recommendation_status': 'exploratory'}}]})
    row, = project_filter_history(tmp_path)['records']
    assert row['stage'] == 'Recommendation eligibility'
    assert row['reason'] == 'Requested motif rule failed.'


def test_missing_external_and_symlinked_evidence_is_not_followed(tmp_path):
    run = tmp_path/'run'; run.mkdir()
    external = save(tmp_path, 'elsewhere.json', {'rows': [native()]})
    save(run, 'artifacts/intake/structured_goal.json', {'filter': {'excluded_path':str(external)}})
    (run/'filtered_no_homopolymer.json').symlink_to(external)
    assert project_filter_history(run) is None
    save(run, 'artifacts/design/tiling/filtered_gc_out_of_range.json', {'rows': []})
    assert project_filter_history(run) is None


def test_region_materialization_rejections_are_not_asos(tmp_path):
    regions = [{'chrom':'chr1', 'start':100, 'end':104, 'reason':'Invalid reference interval'}]
    path = save(tmp_path, 'artifacts/design/rejected_regions.json', {'rows': regions})
    save(tmp_path, 'artifacts/intake/structured_goal.json', {'materialization_summary': {
        'rejected_rows': regions, 'excluded_path':str(path)}})
    assert project_filter_history(tmp_path) is None


def test_report_navigation_adds_history_without_changing_scored_pool(tmp_path):
    from aso_agents.report_explorer_data import project_records
    from aso_agents.report_explorer import pack_data
    candidates = project_records([native(200)], kind='design')
    before = json.dumps(candidates, sort_keys=True)
    save(tmp_path, 'artifacts/design/tiling/filtered_no_homopolymer.json', {'rows': [native()]})
    document = render_goal_intake_html({'meta': {'run_dir':str(tmp_path)},
        'candidate_explorer': candidates, 'overview': {'report_markdown':'# Results\nSaved pool.'}})
    assert 'href="#filtered-candidates">Filtered candidates (1)</a>' in document
    assert json.dumps(candidates, sort_keys=True) == before
    start = document.index('<script type="application/json" class="report-explorer-data">')
    payload = document[start:].split('>',1)[1].split('</script>',1)[0]
    assert json.loads(payload) == pack_data(candidates)
    assert len(json.loads(Packet(document).value)['records']) == 1


def test_all_filtered_run_appears_in_both_production_html_renderers(tmp_path):
    save(tmp_path, 'artifacts/design/tiling/filtered_gc_out_of_range.json', {'rows': [native()]})
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*.json')}
    basic = basic_document('No surviving candidates', '# Results\nNo eligible candidates.', tmp_path)
    design = render_goal_intake_html({'meta': {'run_dir': str(tmp_path)},
        'overview': {'report_markdown': '# Results\nNo eligible candidates.'}})
    for document in [basic, design]:
        assert 'id="filtered-candidates"' in document
        assert len(json.loads(Packet(document).value)['records']) == 1
        assert document.count('class="filter-history-data"') == 1
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*.json')}


def test_large_pool_has_bounded_dom_without_embedding_full_scientific_arrays(tmp_path):
    rows = [native(i, Off_target_positions=['unnecessary bulky evidence']*50) for i in range(2500)]
    save(tmp_path, 'artifacts/design/tiling/filtered_gc_out_of_range.json', {'rows': rows})
    data = project_filter_history(tmp_path)
    document = filter_history_html(data)
    packet = Packet(document)
    assert len(json.loads(packet.value)['records']) == 2500
    assert packet.tr == 26  # Header + first page only.
    assert 'unnecessary bulky evidence' not in document
    assert len(document) < 3_000_000


def test_html_script_escape_and_recorded_text_survive(tmp_path):
    text = '</script><img src=x onerror=alert(1)>'
    save(tmp_path, 'artifacts/candidates/candidate_exclusions.json', {'rows': [native(reason=text)]})
    document = filter_history_html(project_filter_history(tmp_path))
    assert text not in document
    assert json.loads(Packet(document).value)['records'][0]['reason'] == text
