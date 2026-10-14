import json
import pytest
from aso_chemistry import encode_idt_oligo, require_linkage_specification
from aso_agents.chemistry_contracts import compile_chemistry_proposal, chemistry_contract_errors
from aso_recipient_report import attest_record, write_recipient_report, validate_recipient_report
SEQ = 'TACAAATTATGCCTGGAATA'
LINKS = ['PS' if i in [1,*range(6,16),18,19] else 'PO' for i in range(1,20)]
LEGACY = '/52MOErT/*/i2MOErA//i2MOErC//i2MOErA//i2MOErA/A*T*T*A*T*G*/iMe-dC/*/iMe-dC/*T*G*/i2MOErG//i2MOErA//i2MOErA/*/i2MOErT/*/32MOErA/'

def test_legacy_pattern_is_preserved_explicitly():
    assert encode_idt_oligo(SEQ,'5-10-5',linkages=LINKS) == LEGACY
    assert encode_idt_oligo(SEQ,'5-10-5').count('*') == 19

@pytest.mark.parametrize('links', [[], ['PS']*20, ['PS']*18+['unknown'], 'PS', ['PS']*18+[False]])
def test_invalid_pattern_never_falls_back(links):
    with pytest.raises(ValueError):encode_idt_oligo(SEQ,'5-10-5',linkages=links)

def test_native_mixed_selection_is_rejected_before_label_collapse():
    rec={'architecture':{'kind':'gapmer','sugar':'2MOE','backbone':'PS','wing_5prime':5,'dna_gap':10,'wing_3prime':5,'linkages':LINKS},'length':20,'rationale':'Requested chemistry.'}
    with pytest.raises(ValueError, match='do not substitute'):compile_chemistry_proposal({'recommendation':rec})
    rec['architecture']['linkages']=['PS']*19
    assert compile_chemistry_proposal({'recommendation':rec})['recommendation']['linkage_specification']==['PS']*19
    canonical={'recommendation':{'CHEMISTRY':'5-10-5','platform':'gapmer','length':20,'rationale':'Requested profile.','linkage_specification':LINKS}}
    assert any('do not substitute' in error for error in chemistry_contract_errors(canonical))

def record():
    return attest_record('historical',SEQ,'5-10-5',LEGACY,linkages=LINKS,assumption_status='observed_export_intent_unknown',native_rank=1)

def test_contradiction_not_repaired():
    with pytest.raises(ValueError,match='contradicts'):attest_record('x',SEQ,'5-10-5',LEGACY,linkages=['PS']*19,assumption_status='explicit_request')

@pytest.mark.parametrize('view',['candidates.csv','candidates.xlsx','candidates.json','report.md','report.html'])
def test_every_export_is_checked(tmp_path,view):
    p=tmp_path/'report';write_recipient_report(p,[record()],title='Original',context='Computational evidence only.',provenance={})
    assert validate_recipient_report(p)['records_checked']==1
    if view.endswith('.xlsx'):
        from openpyxl import load_workbook
        book=load_workbook(p/view);book.active['G2']='wrong';book.save(p/view)
    elif view.endswith('.json'):
        data=json.loads((p/view).read_text());data['records'][0]['ordering_notation']='wrong';(p/view).write_text(json.dumps(data))
    else:(p/view).write_text((p/view).read_text().replace(LEGACY,'wrong'))
    with pytest.raises(ValueError):validate_recipient_report(p)

def test_original_directory_not_overwritten(tmp_path):
    p=tmp_path/'report';p.mkdir()
    with pytest.raises(FileExistsError):write_recipient_report(p,[record()],title='x',context='',provenance={})


def test_steric_all_ps_and_pmo_pattern_separation():
    arch={'kind':'steric_blocker','sugar':'2MOE','backbone':'PS','wing_5prime':0,'dna_gap':0,'wing_3prime':0,'linkages':['PS']*19}
    payload={'recommendation':{'architecture':arch,'length':20,'rationale':'Fully modified steric blocker.'}}
    assert compile_chemistry_proposal(payload)['recommendation']['linkage_specification']==['PS']*19
    arch['linkages'][1]='PO'
    with pytest.raises(ValueError):compile_chemistry_proposal(payload)
    with pytest.raises(ValueError):encode_idt_oligo('ACGT','steric-pmo',linkages=['PS']*3)


@pytest.mark.parametrize('container', ['direct', 'constraints', 'program'])
def test_actual_native_entrypoints_reject_before_any_model_call(container, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    import run_goal_intake as runtime
    canonical={'CHEMISTRY':'5-10-5','platform':'gapmer','length':20,'linkage_specification':LINKS,'rationale':'Requested profile.'}
    goal={'chemistry':canonical} if container=='direct' else {'constraints':{'chemistry':canonical}} if container=='constraints' else {'design_programs':[{'program_id':'p','chemistry':canonical}]}
    def unexpected(*args, **kwargs):
        raise AssertionError('A model call must not precede linkage rejection.')
    monkeypatch.setattr(pipeline.Runner,'run',unexpected)
    with pytest.raises(ValueError,match='do not substitute'):runtime._resolve_gapmer_chemistry(goal,tile_len=20)
    with pytest.raises(ValueError,match='do not substitute'):pipeline._run_chemistry_agent_for_goal(object(),goal)
    with pytest.raises(ValueError,match='do not substitute'):runtime._normalize_requested_chemistry(canonical)
    with pytest.raises(ValueError,match='do not substitute'):pipeline._canonicalize_chemistry_payload({'recommendation':canonical})


def test_native_all_ps_spec_survives_label_normalization():
    from aso_agents import pipeline_tools as pipeline
    import run_goal_intake as runtime
    canonical={'CHEMISTRY':'5-10-5','platform':'gapmer','length':20,'linkage_specification':['PS']*19,'rationale':'Requested profile.'}
    assert runtime._normalize_requested_chemistry(canonical)['recommendation']['linkage_specification']==['PS']*19
    assert pipeline._canonicalize_chemistry_payload({'recommendation':canonical})['recommendation']['linkage_specification']==['PS']*19
    malformed={**canonical,'linkage_specification':['PS']*18}
    with pytest.raises(ValueError):runtime._normalize_requested_chemistry(malformed)


@pytest.mark.parametrize('notation', [LEGACY.replace('/52MOErT/','/i2MOErT/'), LEGACY.replace('/i2MOErA/','A',1), LEGACY.replace('A*T*T*A*T*G*','/i2MOErA/*T*T*A*T*G*'), LEGACY+'*', LEGACY.replace('/i2MOErC/','/iMOErC/',1)])
def test_original_chemistry_is_independently_parsed(notation):
    with pytest.raises(ValueError):attest_record('x',SEQ,'5-10-5',notation,linkages=LINKS,assumption_status='observed_export_intent_unknown')


def test_attestation_does_not_depend_on_encoder(monkeypatch):
    import aso_chemistry
    monkeypatch.setattr(aso_chemistry,'encode_idt_oligo',lambda *a,**k:'BROKEN')
    result=record()
    assert result['nucleotide_chemistry_5to3'][11:13]==['DNA_methylC','DNA_methylC']
    assert result['linkages_5to3']==LINKS


@pytest.mark.parametrize("backbone", ["PO","mixed-PS-PO","PHOSPHORODIAMIDATE"])
def test_direct_native_backbone_cannot_be_overwritten_by_label(backbone):
    import run_goal_intake as runtime
    with pytest.raises(ValueError,match="do not substitute"):
        runtime._resolve_gapmer_chemistry({"chemistry":{"CHEMISTRY":"5-10-5","backbone":backbone}},tile_len=20)


@pytest.mark.parametrize("architecture", [{"kind":"gapmer","backbone":"PHOSPHORODIAMIDATE"},{"sugar":"2MOE","backbone":"PHOSPHORODIAMIDATE"},{"sugar":"MORPHOLINO","backbone":"PS"}])
def test_nested_typed_backbone_conflicts_are_not_dropped(architecture):
    import run_goal_intake as runtime
    with pytest.raises(ValueError,match="do not substitute"):
        runtime._resolve_gapmer_chemistry({"chemistry":{"CHEMISTRY":"5-10-5","architecture":architecture}},tile_len=20)
