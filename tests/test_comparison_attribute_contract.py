from copy import deepcopy

from aso_agents.comparison_evidence import annotate_comparison_attributes, response_comparison_context
from aso_agents.pipeline_tools import _mixed_genome_build_validation_entry


def test_comparison_ranges_include_exceptions_missing_values_and_measured_zero():
    payload = {'candidate_evidence_by_set':[{'source_type':'generated','top_rows_in_source_order':[
        {'isoform_hit_count':8, 'recorded_scientific_fields':{'selection_score_components':{'phastcons':0, 'phylop':-1}}},
        {'isoform_hit_count':7, 'recorded_scientific_fields':{'selection_score_components':{'phastcons':.8, 'phylop':0}}},
        {'isoform_hit_count':8, 'recorded_scientific_fields':{'selection_score_components':{'phastcons':None, 'phylop':float('nan')}}},
    ]}]}
    result = response_comparison_context(payload)['candidate_evidence_by_set'][0]
    counts = result['recorded_attribute_summaries']['isoform_hit_count']
    assert (counts['minimum'],counts['maximum']) == (7,8)
    assert not counts['all_examined_rows_same_measured_value']
    conservation = result['recorded_attribute_summaries']['phastcons']
    assert conservation['minimum'] == 0 and conservation['unknown_rows'] == 1
    assert result['recorded_attribute_summaries']['phylop']['minimum'] == -1
    assert 'target tile' in result['attribute_definitions']['phastcons_and_phylop']
    assert 'recorded_attribute_summaries' not in payload['candidate_evidence_by_set'][0]


def test_foreign_source_aliases_do_not_block_conversion_but_new_intervals_do():
    goal = {'species':'human','region_constraints':[{'source_type':'explicit_rows','coordinate_space':'genomic',
        'genome_build':'hg19','coordinate_convention':'BED0','allow_coordinate_conversion':True,
        'intervals_bed0_half_open':[{'chrom':'chr2','start':110,'end':120}]}],
        'evaluation_targets':[{'label':'Source feature hg19; one-based inclusive chr2:111-120','start':111,'end':120}],
        'design_programs':[{'objective':'Use the original hg19 chr2:110-120 after validated conversion.',
                            'target_region':'Original hg19 chr2:110-120'}]}
    envelope = {'request_text':'Use hg19 chr2:110-120 on installed hg38.'}
    before = deepcopy(goal)
    validate = lambda: _mixed_genome_build_validation_entry(envelope, {}, {'structured_goal':goal})
    assert validate() is None and goal == before
    goal['design_programs'][0]['target_region'] = 'Also hg19 chr2:190-220'
    assert validate()['issue'] == 'mixed_genome_build_coordinates'
    goal = deepcopy(before); goal['patient_variant'] = {'genome_build':'hg19','chrom':'chr2','start':110}
    assert validate()['issue'] == 'mixed_genome_build_coordinates'
    goal = deepcopy(before); goal['region_constraints'][0]['allow_coordinate_conversion'] = False
    assert validate()['issue'] == 'mixed_genome_build_coordinates'
