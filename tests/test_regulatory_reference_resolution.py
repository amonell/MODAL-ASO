from copy import deepcopy
from pathlib import Path

import pytest

import run_goal_intake as native


def structure(transcript_id='ENST00000999901', strand=1):
    exons = [dict(index=1, start=101, end=160, contains_coding=True,
                  coding_start_genomic=121, coding_end_genomic=140,
                  coding_start=21, coding_end=40)]
    return dict(id=transcript_id.split('.')[0], version=3,
                versioned_id=transcript_id.split('.')[0]+'.3',
                seq_region_name='8', strand=strand, exons_tx=exons,
                reference_source='neutral_local_annotation', reference_annotation='neutral.gtf')


@pytest.mark.parametrize('strand,expected', [(1, [[140, 160]]), (-1, [[100, 120]])])
def test_native_rbp_source_path_resolves_missing_default_before_utr_projection(tmp_path, monkeypatch, strand, expected):
    tx = structure(strand=strand)
    selections = []
    def choose(**kwargs):
        selections.append(kwargs)
        return tx['id']
    monkeypatch.setattr(native, '_pick_canonical_transcript_id', choose)
    monkeypatch.setattr(native, '_fetch_transcript_structure', lambda tid: deepcopy(tx))
    monkeypatch.setattr(native, '_compute_gene_span', lambda goal: {'chrom': 'chr8', 'start0': 100, 'end': 160})
    monkeypatch.setattr(native, '_select_encode_rbp_dataset', lambda **kwargs: {'selected': {
        'file_format': 'bed', 'download_url': 'https://neutral.invalid/sites.bed',
        'file_accession': 'neutral_peak', 'assembly': 'GRCh38'}})
    monkeypatch.setattr(native, '_download_encode_peak_file', lambda url, dest: Path(dest).write_text(
        'chr8\t90\t170\tspanning_peak\t10\t'+('+' if strand == 1 else '-')+'\n'))
    goal = {'species': 'human', 'gene_symbol': 'NEUTRAL', 'target_region': 'utr3'}
    rows, summary = native._materialize_rbp_binding_rows({'rbp': 'FACTOR'}, goal, str(tmp_path))
    assert [row[1:3] for row in rows] == expected
    assert summary['feature_scope']['reference_transcript'] == tx['id']
    original_isoforms = deepcopy(goal['isoform_transcripts'])
    native._ensure_isoform_transcripts_for_target_region(goal)
    assert goal['isoform_transcripts'] == original_isoforms and len(selections) == 1
    assert next(iter(original_isoforms.values()))['metadata']['versioned_id'] == tx['versioned_id']
    # Resolving a default transcript supplies no annotation flag in this fixture.
    # Keep that uncertainty explicit; selecting it must not imply canonical/MANE.
    metadata = next(iter(original_isoforms.values()))['metadata']
    assert metadata['is_canonical'] is None
    assert metadata['is_mane_select'] is None
    assert metadata['is_mane_plus_clinical'] is None


def test_exact_user_lock_precedes_default_and_retains_real_reference_metadata(monkeypatch):
    tx = structure('ENST00000999902.3')
    monkeypatch.setattr(native, '_fetch_transcript_structure', lambda tid: deepcopy(tx))
    monkeypatch.setattr(native, '_pick_canonical_transcript_id', lambda **_: pytest.fail('User lock replaced by default'))
    goal = {'species': 'human', 'gene_symbol': 'NEUTRAL', 'target_region': 'utr3',
            'transcript_policy': 'user_locked', 'transcript_lock': {'value': 'ENST00000999902.3'}}
    final = {}
    native._ensure_isoform_transcripts_for_target_region(goal, final)
    label, entry = native._select_isoform_for_utr(goal['isoform_transcripts'], goal)
    assert label == 'ENST00000999902.3'
    assert entry['metadata']['reference_annotation'] == 'neutral.gtf'
    assert final['isoform_transcripts'] == goal['isoform_transcripts']


@pytest.mark.parametrize('lock', ['ENST00000999902.4', 'unresolved_external_reference'])
def test_unresolved_or_different_version_lock_never_becomes_a_default(monkeypatch, lock):
    monkeypatch.setattr(native, '_fetch_transcript_structure', lambda tid: structure('ENST00000999902.3'))
    monkeypatch.setattr(native, '_pick_canonical_transcript_id', lambda **_: pytest.fail('Unresolved lock became default'))
    goal = {'species': 'human', 'gene_symbol': 'NEUTRAL', 'target_region': 'utr3',
            'transcript_policy': 'user_locked', 'transcript_lock': {'value': lock}}
    native._ensure_isoform_transcripts_for_target_region(goal)
    assert not goal.get('isoform_transcripts')
    existing = {'other': {'transcript_id': 'ENST00000999902', 'metadata': {'version': 3}}}
    assert native._select_isoform_for_utr(existing, goal) == (None, None)


def test_absent_gene_reference_stays_unresolved_without_claiming_no_source_feature(monkeypatch):
    monkeypatch.setattr(native, '_pick_canonical_transcript_id', lambda **_: None)
    goal = {'species': 'human', 'gene_symbol': 'NEUTRAL', 'target_region': 'utr3'}
    native._ensure_isoform_transcripts_for_target_region(goal)
    assert not goal.get('isoform_transcripts')
    assert native._build_utr_bed_rows(goal, 'utr3') == []


def test_no_reference_fetch_for_non_feature_request(monkeypatch):
    monkeypatch.setattr(native, '_pick_canonical_transcript_id', lambda **_: pytest.fail('Unneeded reference selection'))
    goal = {'species': 'human', 'gene_symbol': 'NEUTRAL', 'target_region': 'splice'}
    before = deepcopy(goal)
    native._ensure_isoform_transcripts_for_target_region(goal)
    assert goal == before
