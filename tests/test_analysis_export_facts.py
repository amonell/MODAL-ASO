import csv
from copy import deepcopy
import pytest
from aso_agents.analysis_evidence import validate_recorded_export_facts
from aso_agents.analysis_execution import execute_analysis_program


def payload(rows=None):
    return {"operation": "render_requested_outputs", "declared_inputs": {"completed_run": {
        "evaluation_evidence": [{"rows": rows if rows is not None else [{
            "name": "alpha", "sequence": "ACGT", "length": 4, "gc_fraction": 0.5,
            "orientation_interpretation": {"original_molecule_exact_binding": True},
            "off_target_observed_gene_count": 0, "off_target_count_status": "scoped",
        }]}]}}}


def table(tmp_path, data):
    p=tmp_path/'result.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    return p


@pytest.mark.parametrize('value',['', 'False', '0'])
def test_empty_lookup_cannot_erase_recorded_binding(tmp_path,value):
    p=table(tmp_path,[{'aso_id':'alpha','original_molecule_exact_binding':value}])
    with pytest.raises(ValueError,match='recorded True'):
        validate_recorded_export_facts(payload(),[p])


def test_valid_facts_and_new_calculations_are_preserved(tmp_path):
    p=table(tmp_path,[{'aso_id':'alpha','sequence_5to3':'ACGT','original_molecule_exact_binding':True,
                      'off_target_observed_gene_count':0,'new_calculation':17}])
    before=p.read_bytes();validate_recorded_export_facts(payload(),[p]);assert p.read_bytes()==before


def test_filtered_results_and_aggregate_tables_have_no_invented_row_requirement(tmp_path):
    p=table(tmp_path,[{'group':'subset','count':1}])
    validate_recorded_export_facts(payload(),[p])


def test_conflicting_reference_records_are_not_conflated(tmp_path):
    first=payload()['declared_inputs']['completed_run']['evaluation_evidence'][0]['rows'][0]
    second=deepcopy(first);second['orientation_interpretation']['original_molecule_exact_binding']=False
    p=table(tmp_path,[{'aso_id':'alpha','original_molecule_exact_binding':False}])
    validate_recorded_export_facts(payload([first,second]),[p])


def test_counts_are_exact_but_floating_display_can_round(tmp_path):
    source={'name':'alpha','gc_fraction':1/3,'observed_count':3}
    p=table(tmp_path,[{'aso_id':'alpha','gc_fraction':'0.333','observed_count':'3'}])
    validate_recorded_export_facts(payload([source]),[p])
    p=table(tmp_path,[{'aso_id':'alpha','observed_count':'2.9'}])
    with pytest.raises(ValueError,match='observed_count'):
        validate_recorded_export_facts(payload([source]),[p])


def test_known_fact_error_enters_existing_bounded_code_repair(tmp_path):
    output=tmp_path/'result.csv';calls=[]
    def generate(request):
        calls.append(request)
        value='True' if len(calls)>1 else ''
        if len(calls)>1:
            assert 'recorded True' in request['execution_repair']['error']
        content='aso_id,original_molecule_exact_binding\nalpha,'+value+'\n'
        return {'python_code':"from pathlib import Path\n"+f"Path({str(output)!r}).write_text({content!r})\n"
                +"Path('sandbox_result.json').write_text('{}')\nPath('sandbox_summary.md').write_text('Recorded observations exported.')\n"}
    result=execute_analysis_program(directory=tmp_path,payload=payload(),generate=generate,required_outputs=[output])
    assert result['ok'] and len(result['attempts'])==2 and len(calls)==2
    assert 'alpha,True' in output.read_text()
