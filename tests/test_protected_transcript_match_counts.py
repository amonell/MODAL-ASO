"""Complete transcript-window counts must not be inferred from closest sites."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random

import pytest

from aso_chemistry import encode_idt_oligo
from aso_agents.design_delivery import write_design_panel
from aso_agents.gapmer_selectivity import annotate_preservation, filter_preservation


def annotated(tmp_path, target, references, chemistry='2-5-3'):
    antisense = target.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    row = {'Sequence': target, 'Synthesis': antisense,
           'IDT': encode_idt_oligo(antisense, chemistry)}
    path = tmp_path/'tiles.json'
    path.write_text(json.dumps({'rows': [row]}))
    goal = {'preservation_targets': [
        {'action': 'preserve', 'preservation_target_type': 'isoform', 'isoform_label': label}
        for label in references]}
    annotate_preservation(path, goal, {'beneficial_labels': list(references)}, references, chemistry)
    return path, json.loads(path.read_text())['rows'][0]['gapmer_preservation_assessment']


def oracle(target, reference):
    counts = Counter(sum(a != b for a, b in zip(target, reference[i:i+len(target)]))
                     for i in range(len(reference)-len(target)+1))
    return {str(n): counts[n] for n in range(len(target)+1)}


def test_exact_and_one_substitution_sites_coexist_and_survive_filtering(tmp_path):
    target = 'ACGTACGTGA'
    reference = target + 'GGGG' + target[:-1] + 'C' + 'GGGG' + target
    path, assessment = annotated(tmp_path, target, {'arbitrary_reference.7': reference})
    evidence, = assessment['protected_references']
    expected = oracle(target, reference)
    assert expected['0'] == 2 and expected['1'] >= 1
    assert evidence['ungapped_mismatch_site_counts'] == expected
    assert evidence['ungapped_windows_searched'] == len(reference)-len(target)+1
    assert evidence['transcript_sequence_sha256'] == hashlib.sha256(reference.encode()).hexdigest()
    assert len(evidence['closest_sites']) == 2
    assert all(site['mismatch_count'] == 0 for site in evidence['closest_sites'])
    result = filter_preservation(path)
    rejected = json.loads(Path(result['excluded_path']).read_text())
    assert rejected['rows'][0]['gapmer_preservation_assessment'] == assessment


@pytest.mark.parametrize('seed', range(12))
def test_histogram_matches_independent_exhaustive_count_and_orientation(tmp_path, seed):
    rng = random.Random(seed)
    length = 10 + seed % 3
    target = ''.join(rng.choice('ACGT') for _ in range(length))
    variant = target[:4] + next(base for base in 'ACGT' if base != target[4]) + target[5:]
    reference = ''.join(rng.choice('ACGT') for _ in range(31)) + target + variant
    _, assessment = annotated(tmp_path, target, {'independent.3': reference}, f'2-{length-5}-3')
    evidence, = assessment['protected_references']
    assert evidence['ungapped_mismatch_site_counts'] == oracle(target, reference)
    assert sum(evidence['ungapped_mismatch_site_counts'].values()) == evidence['ungapped_windows_searched']
    assert evidence['target_length_nt'] == len(target)
    assert evidence['transcript_length_nt'] == len(reference)
    assert evidence['minimum_ungapped_mismatches'] == min(
        int(n) for n, count in evidence['ungapped_mismatch_site_counts'].items() if count)


@pytest.mark.parametrize('reference,expected', [
    ('AAAAAAAAAAAA', {'0': 3, '1': 0}),
    ('CAAAAAAAAA', {'0': 0, '1': 1}),
    ('CCCCCCCCCC', {'0': 0, '1': 0}),
])
def test_overlapping_sites_and_measured_zero_bins(tmp_path, reference, expected):
    _, assessment = annotated(tmp_path, 'AAAAAAAAAA', {'different.2': reference})
    evidence, = assessment['protected_references']
    assert {key: evidence['ungapped_mismatch_site_counts'][key] for key in expected} == expected
    assert sum(evidence['ungapped_mismatch_site_counts'].values()) == len(reference)-9


@pytest.mark.parametrize('reference', ['', 'ACGT', 'ACGTACGTGN'])
def test_incomplete_reference_never_exports_zero_bins(tmp_path, reference):
    with pytest.raises(ValueError, match='sequence_incomplete'):
        annotated(tmp_path, 'ACGTACGTGA', {'missing.1': reference})
    assert 'gapmer_preservation_assessment' not in json.loads((tmp_path/'tiles.json').read_text())['rows'][0]


def test_panel_exports_separate_per_reference_counts_and_unknown_legacy_bins(tmp_path):
    target = 'ACGTACGTGA'
    _, assessment = annotated(tmp_path, target, {
        'exact_and_near.4': target + 'GGGG' + target[:-1] + 'C',
        'no_near_sites.2': 'C'*len(target)})
    assessment['protected_references'].append({'transcript_label': 'legacy.1',
        'minimum_ungapped_mismatches': 0, 'closest_sites': []})
    row = {'synthesized_antisense_sequence': target.translate(str.maketrans('ACGT', 'TGCA'))[::-1],
           'target_binding_sequence': target, 'gapmer_preservation_assessment': assessment}
    output = tmp_path/'panel'
    write_design_panel({'rows': [row]}, {}, output)
    delivered = next(csv.DictReader((output/'design_panel.csv').open()))
    assert json.loads(delivered['protected_transcript_exact_site_counts']) == {
        'exact_and_near.4': 1, 'no_near_sites.2': 0, 'legacy.1': None}
    one = json.loads(delivered['protected_transcript_one_mismatch_site_counts'])
    assert one['exact_and_near.4'] >= 1 and one['no_near_sites.2'] == 0 and one['legacy.1'] is None
    text = (output/'design_panel.md').read_text()
    assert '| Exact sites | One-mismatch sites | Windows searched |' in text
    assert '| legacy.1 | unknown | unknown | unknown |' in text
    assert 'distinct transcript start positions' in text
    assert json.loads(delivered['gapmer_preservation_assessment']) == assessment


def test_annotations_fallback_and_no_assessment_are_not_zero(tmp_path):
    target = 'ACGTACGTGA'
    _, assessment = annotated(tmp_path, target, {'reference.1': target})
    rows = [{'synthesized_antisense_sequence': 'TCACGTACGT',
             'source_annotations': {'gapmer_preservation_assessment': assessment}},
            {'synthesized_antisense_sequence': 'G'*10}]
    output = tmp_path/'fallback'
    write_design_panel({'rows': rows}, {}, output)
    delivered = list(csv.DictReader((output/'design_panel.csv').open()))
    assert json.loads(delivered[0]['protected_transcript_exact_site_counts']) == {'reference.1': 1}
    assert json.loads(delivered[1]['protected_transcript_exact_site_counts']) == {}


@pytest.mark.parametrize('value', [False, '0', -1, None])
def test_invalid_recorded_counts_are_not_presented_as_measured_zero(value):
    from aso_agents.design_delivery import _protected_site_count
    assert _protected_site_count({'ungapped_mismatch_site_counts': {'0': value}}, 0) is None
