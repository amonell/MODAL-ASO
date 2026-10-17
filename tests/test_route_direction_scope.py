from copy import deepcopy
from aso_agents import pipeline_tools as p


def fixture():
    site={'chrom':'chr7','position':310,'site':'donor','strand':'+',
          'delta_direction':'decrease','delta_per_isoform':{'short_form':'decrease'},
          'present_in':['short_form'],'absent_in':['long_form'],'covers_all_isoforms':False}
    program={'program_id':'restore_segment','modality':'steric','intent':'splice_modulation',
             'isoform_label':'short_form','desired_delta':'increase','desired_effect':'inclusion','site_kind':'donor'}
    goal={'splice_site_plan':[site],
          'splice_isoform_objectives':[{'objective_scope':'specific_isoform','isoform_label':'short_form','desired_abundance':'decrease'}]}
    return goal,program,site


def test_event_inclusion_and_isoform_reduction_do_not_conflict():
    goal,program,site=fixture();original=deepcopy(program)
    scoped=p._scope_exploratory_sites_to_program(structured_goal=goal,site_entries=[site],base_program=program)
    assert scoped==[site] and program==original
    derived,meta=p._derive_exploratory_steric_programs(structured_goal=goal,base_program=program)
    assert derived[0]['anchor_sites']==['donor@310']
    assert derived[0]['desired_effect']=='inclusion'
    assert derived[0]['desired_delta']=='increase'


def test_other_identity_and_conflicting_outcomes_are_not_admitted():
    goal,program,site=fixture()
    other={**site,'delta_per_isoform':{'other_form':'decrease'},'present_in':['other_form']}
    assert p._scope_exploratory_sites_to_program(structured_goal=goal,site_entries=[other],base_program=program)==[]
    goal['evaluation_targets']=[{'target_type':'isoform','isoform_label':'short_form','desired_delta':'increase'}]
    assert p._scope_exploratory_sites_to_program(structured_goal=goal,site_entries=[site],base_program=program)==[]


def test_structural_event_does_not_override_isoform_abundance():
    goal,program,site=fixture()
    goal['splice_isoform_objectives'].append({'objective_scope':'default_exon_context','isoform_label':'short_form','desired_abundance':'increase','exon_label':3})
    goal['evaluation_targets']=[{'target_type':'exon','isoform_label':'short_form','desired_delta':'increase','exon_label':3}]
    assert p._program_isoform_abundance_directions(goal,program)=={'decrease'}
