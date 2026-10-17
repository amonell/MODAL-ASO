import csv
import json
from pathlib import Path

import pytest
from aso_agents import pipeline_tools as pipeline
from aso_agents.requested_artifacts import copy_native_export


def test_scoped_export_filters_recorded_programs_without_rewriting_evidence(tmp_path):
    source = tmp_path/'source.csv'
    source.write_text('program_id,source_modality,rank,sequence,unknown\n'
                      'alpha,gapmer,0001,AACG,\n'
                      'beta,steric,0002,CCGT,\n'
                      'gamma,gapmer,0003,TTGC,\n')
    before = source.read_bytes()
    destination = tmp_path/'output.csv'
    receipt = copy_native_export(source, destination,
        scope={'program_ids':['alpha','beta'], 'modalities':['gapmer']})
    assert list(csv.DictReader(destination.open())) == [
        {'program_id':'alpha','source_modality':'gapmer','rank':'0001','sequence':'AACG','unknown':''}]
    assert receipt['row_count'] == 1 and receipt['source_row_count'] == 3
    assert source.read_bytes() == before


def test_scope_does_not_replace_absent_branch_with_other_branch(tmp_path):
    source=tmp_path/'source.csv'; source.write_text('program_id,source_modality,sequence\na,gapmer,AACG\n')
    destination=tmp_path/'output.csv'
    receipt=copy_native_export(source,destination,scope={'modalities':['steric']})
    assert list(csv.DictReader(destination.open())) == []
    assert receipt['row_count'] == 0


def test_steric_scope_includes_native_splice_route_rows_without_relabeling(tmp_path):
    source=tmp_path/'source.csv';source.write_text('program_id,source_modality,sequence\na,gapmer,AACG\nb,splice_mod,CCGT\n')
    destination=tmp_path/'output.csv'
    copy_native_export(source,destination,scope={'modalities':['steric']})
    assert list(csv.DictReader(destination.open())) == [
        {'program_id':'b','source_modality':'splice_mod','sequence':'CCGT'}]


def test_scoped_export_rejects_missing_identity_and_unsupported_format(tmp_path):
    source=tmp_path/'source.csv';source.write_text('sequence\nAACG\n')
    with pytest.raises(ValueError,match='scope_column_missing'):
        copy_native_export(source,tmp_path/'output.csv',scope={'program_ids':['a']})
    source=tmp_path/'source.md';source.write_text('Unscoped aggregate report')
    with pytest.raises(ValueError,match='scope_unsupported'):
        copy_native_export(source,tmp_path/'output.md',scope={'modalities':['steric']})


def test_final_output_scope_survives_normalization_and_delivery(tmp_path,monkeypatch):
    source=tmp_path/'source.csv';source.write_text('program_id,source_modality,sequence\na,gapmer,AACG\nb,steric,CCGT\n')
    outputs=[{'kind':'design_panel_csv','task_id':'design','path':name+'.csv',
              'source_artifact':'design_panel_csv','source_scope':{'modalities':[name]}}
             for name in ['gapmer','steric']]
    normalized=pipeline._normalize_final_outputs(outputs,'design')
    assert [spec.get('source_scope') for spec in normalized] == [spec['source_scope'] for spec in outputs]
    monkeypatch.setattr(pipeline,'_native_export_catalog',lambda *a: {'design_panel_csv':str(source)})
    state={'request_envelope':{'tasks':[{'task_id':'design','task_type':'design_new_asos'}],
                               'final_outputs':normalized}}
    result=pipeline._materialize_requested_final_outputs(state,str(tmp_path))
    assert result['ok']
    assert [r['sequence'] for r in csv.DictReader((tmp_path/'gapmer.csv').open())] == ['AACG']
    assert [r['sequence'] for r in csv.DictReader((tmp_path/'steric.csv').open())] == ['CCGT']
