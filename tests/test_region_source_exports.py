from copy import deepcopy
from pathlib import Path
import pytest
from aso_agents.region_source_exports import export_region_sources
from aso_agents.report_explainer import _collect_important_artifacts


def test_exact_source_bed_does_not_substitute_expanded_or_merged_search_windows(tmp_path):
    rows = [['chr4',100,120,'site_A',0,'-'], ['chr4',110,140,'site_B',0,'-']]
    goal = {'materialized_region_constraints':{'global':[{'constraint_id':'sites','coordinate_space':'genomic','materialized_rows':rows}]},
            'steric_bed':{'rows':[['chr4',81,159,'search',0,'-']]}}
    before=deepcopy(goal); result=export_region_sources(goal,tmp_path)
    assert goal==before
    path=Path(result['records'][0]['path'])
    assert path.read_text()=='chr4\t100\t120\tsite_A\t0\t-\nchr4\t110\t140\tsite_B\t0\t-\n'
    assert any(item['path']==str(path.relative_to(tmp_path)) for item in _collect_important_artifacts(tmp_path))


def test_transcript_coordinates_and_distinct_program_scopes_are_labeled(tmp_path):
    c={'coordinate_space':'cdna','materialized_rows':[['TX.3',5,19,'site',0,'+']]}
    goal={'materialized_region_constraints':{'global':[],'by_program':{'one':[c],'two':[c]}}}
    records=export_region_sources(goal,tmp_path)['records']
    assert [r['program_id'] for r in records]==['one','two']
    assert all(r['coordinate_space']=='cdna' for r in records)
    assert len({r['path'] for r in records})==2


def test_invalid_source_is_never_written_as_valid_bed(tmp_path):
    with pytest.raises(ValueError):
        export_region_sources({'materialized_region_constraints':{'global':[{'materialized_rows':[['chr4',100,99]]}]}},tmp_path)
