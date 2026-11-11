from copy import deepcopy
import hashlib
import json
import xml.etree.ElementTree as ET

import pytest
import requests

from aso_agents import curated_exon_nomenclature as naming
from aso_agents.report_exon_names import project_exon_names, bind_site_exon_name, ordinal_word
from aso_agents.report_explorer_data import site_name


def fixture(*, negative=False):
    root = ET.fromstring('''<lrg><fixed_annotation><id>LRG_99999</id><organism taxon="9606">Human</organism>
      <transcript name="t1"/></fixed_annotation><updatable_annotation>
      <annotation_set type="lrg"><mapping coord_system="GRCh38.p14" other_name="3" other_id="NC_000003.12" type="main_assembly">
        <mapping_span lrg_start="1" lrg_end="200" other_start="1001" other_end="1200" strand="1"/>
      </mapping></annotation_set>
      <annotation_set type="ensembl"><features><gene><transcript accession="ENST00000000001.2" fixed_id="t1"/>
      </gene></features></annotation_set>
      <annotation_set type="ncbi"><features><gene><transcript accession="NM_999999.4" fixed_id="t1"/>
      </gene></features></annotation_set>
      <annotation_set type="community"><fixed_transcript_annotation name="t1">
        <other_exon_naming description="Published exon names"><url>https://example.org/names</url></other_exon_naming>
      </fixed_transcript_annotation></annotation_set></updatable_annotation></lrg>''')
    if negative:
        root.find('.//mapping_span').set('strand', '-1')
    ET.SubElement(root.find('.//transcript[@accession="ENST00000000001.2"]'), 'db_xref', source='RefSeq', accession='NM_999999.4')
    exons = []
    for index, (a, b, label) in enumerate([(1, 20, '1'), (51, 70, '2a'), (101, 120, '2')], 1):
        for transcript in root.findall('.//transcript'):
            exon = ET.SubElement(transcript, 'exon', label=str(index))
            ET.SubElement(exon, 'coordinates', coord_system='LRG_99999', start=str(a), end=str(b), strand='1')
        exon = ET.SubElement(root.find('.//other_exon_naming'), 'exon')
        ET.SubElement(exon, 'coordinates', coord_system='LRG_99999', start=str(a), end=str(b), strand='1')
        ET.SubElement(exon, 'label').text = label
        start, end = (1201-b, 1201-a) if negative else (1000+a, 1000+b)
        exons.append({'index': index, 'start': start, 'end': end, 'seq_region_name': 'chr3'})
    entry = {'transcript_id': 'ENST00000000001', 'chrom': 'chr3', 'strand': '-' if negative else '+',
             'exons_tx': exons, 'metadata': {'versioned_id': 'ENST00000000001.2', 'gene_symbol': 'TEST',
                                           'assembly_name': 'GRCh38', 'species': 'homo_sapiens'}}
    return root, entry


def parse(root, entry):
    context = naming.reference_context(entry)
    return naming.parse_lrg_nomenclature(ET.tostring(root, encoding='unicode'),
        transcript_id=context['transcript_id'], transcript_exons=entry['exons_tx'], context=context)


def mock_sources(monkeypatch, root):
    calls = []
    def read(url, timeout):
        calls.append(url)
        text = ('#tax_id\tSymbol\tLRG\tRNA\tRSG\tt\n9606\tTEST\tLRG_99999\tNM_999999.4\tNG_999999.1\tt1\n'
                if url == naming.INDEX_URL else ET.tostring(root, encoding='unicode'))
        return text, hashlib.sha256(text.encode()).hexdigest()
    monkeypatch.setattr(naming, '_read', read)
    return calls


@pytest.mark.parametrize('negative', [False, True])
def test_exact_names_transcript_order_and_strand(monkeypatch, negative):
    root, entry = fixture(negative=negative)
    calls = mock_sources(monkeypatch, root)
    receipt = naming.ensure_entry_nomenclature(entry)
    assert receipt['status'] == 'verified'
    assert receipt['ordered_biological_labels'] == ['1', '2a', '2']
    assert naming.resolve_curated_label(receipt, 2)['resolved_transcript_ordinal'] == 3
    assert naming.resolve_curated_label(receipt, '2a')['resolved_transcript_ordinal'] == 2
    assert naming.resolve_curated_label(receipt, 'EXON 2')['resolved_transcript_ordinal'] == 3
    assert naming.resolve_curated_label(receipt, 7)['status'] == 'unresolved'
    assert naming.ensure_entry_nomenclature(entry) == receipt and len(calls) == 2
    assert 'sequence' not in receipt and len(json.dumps(receipt)) < 4000


@pytest.mark.parametrize('change', ['version', 'chromosome', 'assembly', 'strand', 'one_base', 'order', 'ordinal', 'overlap'])
def test_incompatible_reference_cannot_acquire_names(change):
    root, entry = fixture()
    if change == 'version': entry['metadata']['versioned_id'] = 'ENST00000000001.3'
    if change == 'chromosome': entry['chrom'] = 'chr4'
    if change == 'assembly': entry['metadata']['assembly_name'] = 'GRCh37'
    if change == 'strand': entry['strand'] = '-'
    if change == 'one_base': entry['exons_tx'][1]['end'] += 1
    if change == 'order': entry['exons_tx'][0]['index'], entry['exons_tx'][1]['index'] = 2, 1
    if change == 'ordinal': entry['exons_tx'][1]['index'] = 1
    if change == 'overlap': entry['exons_tx'][1]['start'] = entry['exons_tx'][0]['start']
    with pytest.raises(ValueError): parse(root, entry)


def test_reference_aliases_are_explicit_and_versioned():
    root, entry = fixture()
    entry['metadata']['assembly_name'] = 'hg38'
    assert parse(root, entry)['lrg_id'] == 'LRG_99999'
    entry['transcript_id'] = 'NM_999999.4'
    assert parse(root, entry)['source_transcript_id'] == 'NM_999999.4'
    entry['transcript_id'] = 'NM_999999.3'
    with pytest.raises(ValueError): parse(root, entry)


@pytest.mark.parametrize('change', ['conflict', 'duplicate', 'partial', 'no_attribution', 'wrong_scope'])
def test_ambiguous_or_unattributed_naming_remains_unresolved(change):
    root, entry = fixture()
    other = root.find('.//other_exon_naming')
    if change == 'conflict':
        second = deepcopy(other); second.findall('exon/label')[-1].text = '3'
        root.find('.//fixed_transcript_annotation').append(second)
    if change == 'duplicate': other.findall('exon/label')[-1].text = '1'
    if change == 'partial': other.remove(other.findall('exon')[-1])
    if change == 'no_attribution': other.remove(other.find('url'))
    if change == 'wrong_scope': root.find('.//transcript[@accession="ENST00000000001.2"]').set('fixed_id', 't2')
    with pytest.raises(ValueError): parse(root, entry)


def test_plain_curated_numbering_is_verified_without_claiming_legacy_names(monkeypatch):
    root, entry = fixture()
    parent = root.find('.//fixed_transcript_annotation'); parent.remove(parent.find('other_exon_naming'))
    mock_sources(monkeypatch, root)
    receipt = naming.ensure_entry_nomenclature(entry)
    result = naming.resolve_curated_label(receipt, 2)
    assert result['status'] == 'verified_curated_mapping' and result['resolved_transcript_ordinal'] == 2
    assert receipt['naming_system'] == 'LRG_99999 exon numbering'


def test_haplotype_and_assembly_mappings_are_not_selected_by_order():
    root, entry = fixture()
    parent = root.find('.//annotation_set[@type="lrg"]')
    haplotype = deepcopy(parent.find('mapping'))
    haplotype.attrib.update(other_name='3p', other_id='NT_999999.1', type='haplotype')
    haplotype.find('mapping_span').set('other_start', '100')
    haplotype.find('mapping_span').set('other_end', '299')
    parent.insert(0, haplotype)
    assert parse(root, entry)['assembly_mapping']['other_id'] == 'NC_000003.12'
    parent.remove(parent.findall('mapping')[1])
    with pytest.raises(ValueError): parse(root, entry)


@pytest.mark.parametrize('negative', [False, True])
@pytest.mark.parametrize('kind', ['other_ins', 'lrg_ins'])
def test_intronic_indels_are_applied_to_downstream_coordinates(negative, kind):
    root, entry = fixture(negative=negative)
    span = root.find('.//mapping_span')
    delta = 3 if kind == 'other_ins' else -3
    span.set('other_end', str(1200+delta))
    ET.SubElement(span, 'diff', type=kind, lrg_start='80', lrg_end='81' if kind == 'other_ins' else '82',
                  other_start='1080', other_end='1082' if kind == 'other_ins' else '1081')
    for i, exon in enumerate(entry['exons_tx']):
        shift = (delta if i < 2 else 0) if negative else (delta if i == 2 else 0)
        exon['start'] += shift; exon['end'] += shift
    assert parse(root, entry)['ordered_biological_labels'] == ['1', '2a', '2']


@pytest.mark.parametrize('kind', ['other_ins', 'lrg_ins', 'unknown_gap'])
def test_exonic_or_unknown_gaps_cannot_be_treated_as_linear_mappings(kind):
    root, entry = fixture()
    span = root.find('.//mapping_span')
    span.set('other_end', '1201')
    ET.SubElement(span, 'diff', type=kind, lrg_start='60', lrg_end='61', other_start='1061', other_end='1061')
    with pytest.raises(ValueError): parse(root, entry)


def test_sequence_mismatch_does_not_change_coordinate_identity():
    root, entry = fixture()
    ET.SubElement(root.find('.//mapping_span'), 'diff', type='mismatch', lrg_start='60', lrg_end='60',
                  other_start='1060', other_end='1060')
    assert parse(root, entry)['ordered_biological_labels'] == ['1', '2a', '2']


@pytest.mark.parametrize('failure', ['timeout', 'malformed_xml', 'index_schema', 'missing_record'])
def test_source_failure_preserves_a_readable_ordinal_fallback(monkeypatch, failure):
    root, entry = fixture()
    def read(url, timeout):
        if failure == 'timeout': raise requests.Timeout('source timed out')
        if failure == 'index_schema': return 'unexpected schema', 'a'*64
        if url == naming.INDEX_URL:
            return '#tax_id\tSymbol\tLRG\tRNA\tRSG\n9606\tTEST\t'+('' if failure == 'missing_record' else 'LRG_99999')+'\tNM_999999.4\tNG_999999.1\n', 'a'*64
        return '<broken', 'b'*64
    monkeypatch.setattr(naming, '_read', read)
    receipt = naming.ensure_entry_nomenclature(entry)
    assert receipt['status'] == 'unresolved'
    view = project_exon_names(entry)
    assert view['exons'][2]['label'] == 'third annotated exon'
    assert not view['exon_naming']['verified']


@pytest.mark.parametrize('field,value', [('species', 'mus_musculus'), ('assembly_name', None), ('species', None)])
def test_incomplete_or_nonhuman_identity_needs_no_download(monkeypatch, field, value):
    _, entry = fixture(); entry['metadata'][field] = value
    monkeypatch.setattr(naming, '_read', lambda *_: pytest.fail('Unexpected download'))
    assert naming.ensure_entry_nomenclature(entry)['status'] == 'not_attempted'


@pytest.mark.parametrize('change', ['version', 'coordinates', 'strand'])
def test_saved_receipt_invalidated_by_reference_changes(monkeypatch, change):
    root, entry = fixture(); calls = mock_sources(monkeypatch, root)
    naming.ensure_entry_nomenclature(entry)
    assert project_exon_names(entry)['exon_naming']['verified']
    if change == 'version': entry['metadata']['versioned_id'] = 'ENST00000000001.3'
    if change == 'coordinates': entry['exons_tx'][2]['end'] += 1
    if change == 'strand': entry['strand'] = '-'
    assert not project_exon_names(entry)['exon_naming']['verified']
    assert naming.ensure_entry_nomenclature(entry)['status'] == 'unresolved'
    assert len(calls) == 4


@pytest.mark.parametrize('negative', [False, True])
def test_report_site_names_require_the_attested_reference_and_boundary(monkeypatch, negative):
    root, entry = fixture(negative=negative); mock_sources(monkeypatch, root)
    naming.ensure_entry_nomenclature(entry); view = project_exon_names(entry)
    e = entry['exons_tx'][2]
    site = {'reference_binding_status': 'resolved_catalog_identity', 'reference_transcript_id': 'ENST00000000001.2',
            'site_kind': 'acceptor', 'exon_index': 3,
            'acceptor': {'position': e['end'] if negative else e['start'], 'chrom': 'chr3', 'strand': entry['strand']}}
    bound = bind_site_exon_name(site, entry, view['exons'])
    assert site_name('acceptor@1101', {'acceptor@1101': {**site, **bound}}) == 'Exon 2 acceptor'
    for field, value in [('reference_transcript_id', 'ENST00000000001.1'), ('exon_index', 2)]:
        bad = {**site, field: value}
        assert bind_site_exon_name(bad, entry, view['exons'])['verified_exon_label'] is None
    for field, value in [('chrom', 'chr4'), ('strand', '+' if negative else '-'), ('position', 9999)]:
        bad = {**site, 'acceptor': {**site['acceptor'], field: value}}
        assert bind_site_exon_name(bad, entry, view['exons'])['verified_exon_label'] is None
    entry['metadata'].pop('curated_exon_nomenclature')
    fallback = bind_site_exon_name(site, entry, project_exon_names(entry)['exons'])
    assert site_name('acceptor@1101', {'acceptor@1101': {**site, **fallback}}) == 'Acceptor of the third annotated exon'


def test_alternate_names_drive_intake_and_evaluation_without_model_calls(monkeypatch):
    from aso_agents import exon_nomenclature
    from aso_agents.evaluation_feature_identity import resolve_evaluation_target_identities
    import run_goal_intake
    root, entry = fixture(); mock_sources(monkeypatch, root)
    monkeypatch.setattr(exon_nomenclature, '_run_schema_agent', lambda *_args, **_kwargs: pytest.fail('Unexpected model call'))
    assert run_goal_intake._resolve_authoritative_transcript_exon_ordinal(entry, 2) == 3
    result = resolve_evaluation_target_identities([{'exon_label': 2, 'transcript_id': 'ENST00000000001.2'}],
        {'test': entry}, None, resolver=run_goal_intake._cached_authoritative_exon_nomenclature_mapping)[0]
    assert result['exon_identity_resolution']['status'] == 'verified_biological_label'
    assert result['exon_identity_resolution']['start'] == 1101


def test_display_ordinals_have_correct_suffixes():
    assert [ordinal_word(n) for n in [8, 21, 22, 23, 24, 111, 112, 113]] == [
        'eighth', '21st', '22nd', '23rd', '24th', '111th', '112th', '113th']


def test_refseq_alias_requires_an_explicit_pair_and_matching_structure(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    result = naming.resolve_curated_label(naming.ensure_entry_nomenclature(entry), 2)
    assert result['refseq_accession'] == 'NM_999999.4'
    assert result['ensembl_exon_lengths'] == result['refseq_exon_lengths'] == [20, 20, 20]
    from aso_agents.design_delivery import _resolved_accession_and_record
    entry['metadata']['exon_nomenclature_resolutions'] = {'2': result}
    accession, matched = _resolved_accession_and_record({'transcript_id': 'NM_999999.4',
        'isoform_transcripts': {'ENST00000000001': entry}})
    assert matched is entry
    root.find('.//transcript[@accession="NM_999999.4"]/exon/coordinates').set('end', '21')
    assert parse(root, entry)['refseq_accession'] is None
    root.find('.//transcript[@accession="NM_999999.4"]/exon/coordinates').set('end', '20')
    source = root.find('.//transcript[@accession="ENST00000000001.2"]')
    source.remove(source.find('db_xref'))
    assert parse(root, entry)['refseq_accession'] is None


def test_source_outage_is_bounded_across_transcripts(monkeypatch):
    naming._download.cache_clear()
    calls = []
    def get(*args, **kwargs):
        calls.append(kwargs['timeout'])
        raise requests.Timeout('test')
    monkeypatch.setattr(requests, 'get', get)
    monkeypatch.setattr(naming.time, 'time', lambda: 100)
    for _ in range(3):
        _, entry = fixture()
        assert naming.ensure_entry_nomenclature(entry)['status'] == 'unresolved'
    assert calls == [15]
    naming._download.cache_clear()


def test_modified_transcript_cannot_inherit_curated_names(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    naming.ensure_entry_nomenclature(entry)
    entry['explicit_target_modified'] = True
    assert not project_exon_names(entry)['exon_naming']['verified']
    assert naming.ensure_entry_nomenclature(entry)['status'] == 'not_attempted'


def test_stale_mapping_is_rejected_by_both_design_and_evaluation(monkeypatch):
    from aso_agents.evaluation_feature_identity import resolve_evaluation_target_identities
    from aso_agents.gapmer_selectivity import _ordinal
    root, entry = fixture(); mock_sources(monkeypatch, root)
    receipt = naming.ensure_entry_nomenclature(entry)
    entry['metadata']['exon_nomenclature_resolutions'] = {'2': naming.resolve_curated_label(receipt, 2)}
    assert _ordinal(entry, 2) == 3
    entry['exons_tx'][2]['start'] += 1
    assert _ordinal(entry, 2) == 2
    result = resolve_evaluation_target_identities([{'exon_label': 2, 'transcript_id': 'ENST00000000001.2'}],
                                                  {'test': entry}, None)[0]['exon_identity_resolution']
    assert not result['biological_label_verified']
    assert result['status'] == 'annotation_ordinal_hypothesis'


def test_report_projection_uses_curated_names_without_network_or_source_edits(monkeypatch, tmp_path):
    from test_report_explorer import run_fixture, save
    from aso_agents.report_explorer_data import project_run
    from aso_agents.report_explorer import explorer_html
    root, entry = fixture(); mock_sources(monkeypatch, root)
    naming.ensure_entry_nomenclature(entry)
    monkeypatch.setattr(naming, '_read', lambda *_args: pytest.fail('Report rendering must work offline'))
    goal, raw, pool = run_fixture(tmp_path)
    goal['isoform_transcripts'] = {'ENST00000000001': entry}
    for row in [*raw, *pool]:
        row.update(Start=row['Start']+1000, End=row['End']+1000, Chrom='chr3')
        row['splice_site_deltas'][0].update(canonical_label='acceptor@1101', exon_index=3,
            reference_transcript_id='ENST00000000001.2', acceptor={'chrom':'chr3', 'position':1101, 'strand':'+'})
    pool[0]['route_label'] = 'acceptor@1101'
    save(tmp_path, 'artifacts/intake/structured_goal.json', goal)
    save(tmp_path, 'artifacts/design/steric_jobs/j/scored_tiles.json', {'rows':raw})
    save(tmp_path, 'artifacts/candidates/candidate_assessments.json', {'rows':pool})
    before = {str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}
    data = project_run(tmp_path)
    assert data['contexts'][0]['label'].endswith('Exon 2 acceptor')
    assert data['contexts'][0]['exons'][1]['label'] == 'Exon 2a'
    document = explorer_html(data)
    assert 'Exon names and reference' in document
    assert 'Published exon names' in document and 'LRG_99999' in document
    assert before == {str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}
    entry['metadata'].pop('curated_exon_nomenclature')
    fallback = project_run(tmp_path, goal=goal)
    assert fallback['contexts'][0]['label'].endswith('Acceptor of the third annotated exon')
    assert [(r['aso'], r['effect'], r['predictions'], r['intervals']) for r in data['records']] == [
           (r['aso'], r['effect'], r['predictions'], r['intervals']) for r in fallback['records']]


def test_public_resolver_uses_structured_evidence_without_a_model(monkeypatch):
    import asyncio
    from aso_agents import exon_nomenclature
    root, entry = fixture(); mock_sources(monkeypatch, root)
    monkeypatch.setattr(exon_nomenclature, '_run_schema_agent', lambda *_args, **_kwargs: pytest.fail('Unexpected model call'))
    context = naming.reference_context(entry)
    result = asyncio.run(exon_nomenclature.resolve_transcript_exon_nomenclature(
        transcript_id=context['transcript_id'], requested_exon_label=2,
        transcript_exons=entry['exons_tx'], reference_context=context))
    assert result['resolved_transcript_ordinal'] == 3
    assert result['status'] == 'verified_curated_mapping'


def test_run_persistence_records_names_for_reports_without_an_exon_request(monkeypatch, tmp_path):
    import run_goal_intake
    from aso_agents.run_layout import resolve_run_path
    root, entry = fixture(); mock_sources(monkeypatch, root)
    payload = {'structured_goal': {'isoform_transcripts': {'test': entry}}}
    run_goal_intake._persist_outputs(str(tmp_path), 'An ordinary report request.', payload, [])
    saved = json.loads(resolve_run_path(tmp_path, 'structured_goal.json').read_text())
    receipt = saved['isoform_transcripts']['test']['metadata']['curated_exon_nomenclature']
    assert receipt['status'] == 'verified'
    assert receipt['ordered_biological_labels'] == ['1', '2a', '2']


@pytest.mark.parametrize('explicit_fixed_id', [True, False])
def test_changed_transcript_ends_keep_exact_internal_names(monkeypatch, explicit_fixed_id):
    root, entry = fixture()
    transcript = root.find('.//transcript[@accession="ENST00000000001.2"]')
    if not explicit_fixed_id:
        transcript.attrib.pop('fixed_id')
    transcript.findall('exon/coordinates')[0].set('start', '2')
    transcript.findall('exon/coordinates')[-1].set('end', '121')
    entry['exons_tx'][0]['start'] += 1
    entry['exons_tx'][-1]['end'] += 1
    mock_sources(monkeypatch, root)
    receipt = naming.ensure_entry_nomenclature(entry)
    assert receipt['status'] == 'verified'
    assert receipt['ordered_biological_labels'] == [None, '2a', None]
    assert (receipt['named_exon_count'], receipt['total_exon_count']) == (1, 3)
    assert naming.resolve_curated_label(receipt, '2a')['resolved_transcript_ordinal'] == 2
    assert naming.resolve_curated_label(receipt, 2)['status'] == 'unresolved'
    names = project_exon_names(entry)
    assert [e['label'] for e in names['exons']] == ['first annotated exon', 'Exon 2a', 'third annotated exon']
    assert '1 of 3 exons' in names['exon_naming']['description']
    assert receipt['refseq_accession'] is None
