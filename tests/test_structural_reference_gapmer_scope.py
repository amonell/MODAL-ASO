from aso_agents import pipeline_tools as p
import pytest
from aso_chemistry import require_modality_chemistry


def test_exon_context_reference_is_not_a_whole_transcript_knockdown_target():
    goal={'splice_isoform_objectives':[
        {'objective_scope':'default_exon_context','transcript_id':'TX_REF','isoform_label':'TX_REF',
         'exon_label':'3b','desired_abundance':'decrease'},
        {'objective_scope':'aggregate_class','isoform_label':'containing_feature',
         'aggregate_class':{'basis':'exon_presence','exon_label':'3b','presence':'included'},'desired_abundance':'decrease'},
        {'objective_scope':'specific_transcript','transcript_id':'NM_123456.7','isoform_label':'NM_123456.7','desired_abundance':'decrease'}]}
    assert [x['isoform_label'] for x in p._splice_switching_gapmer_targets(goal,require_direct_signal=True)]==['NM_123456.7']


def test_derived_knockdown_carries_its_own_mechanism(monkeypatch):
    goal={'intent':'splice_modulation','chemistry':'steric-hindrance',
          'splice_isoform_objectives':[{'isoform_label':'NM_123456.7','desired_abundance':'decrease'}]}
    monkeypatch.setattr(p,'_splice_switch_gapmer_support_policy',lambda *a:{'enabled':True,'target_isoforms':['NM_123456.7']})
    programs,plan=p._synthesize_splice_switch_gapmer_programs(goal,[])
    assert len(programs)==1
    assert programs[0]['chemistry']=='gapmer' and programs[0]['chemistry_policy']=='select'
    assert goal['chemistry']=='steric-hindrance'
    goal['chemistry']='5-12-5'
    programs,_=p._synthesize_splice_switch_gapmer_programs(goal,[])
    assert programs[0]['chemistry']=='5-12-5'
    goal.update(chemistry=None,chemistry_policy='leave_unspecified')
    programs,_=p._synthesize_splice_switch_gapmer_programs(goal,[])
    assert programs[0]['chemistry']=='unspecified' and programs[0]['chemistry_policy']=='leave_unspecified'


def test_known_architecture_must_match_design_mechanism():
    require_modality_chemistry('gapmer','5-12-5')
    require_modality_chemistry('steric','steric-pmo')
    require_modality_chemistry('gapmer','unspecified')
    with pytest.raises(ValueError,match='chemistry/mechanism conflict'):
        require_modality_chemistry('gapmer','steric-hindrance')
    with pytest.raises(ValueError,match='chemistry/mechanism conflict'):
        require_modality_chemistry('steric','5-10-5')
