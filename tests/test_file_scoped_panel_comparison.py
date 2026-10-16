import csv

from aso_agents import pipeline_tools as pipeline


def write_panel(path, rows, with_sets=False):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate_id', 'molecule_5to3', 'chemistry'] + (['set_id'] if with_sets else []))
        writer.writerows(rows)


def collect(context, evaluated=None):
    return pipeline._collect_aso_compare_sets(
        state={}, task={'task_id': 'compare', 'context': context}, task_context=context,
        request_envelope=context, evaluated_payload=evaluated or {})


def test_files_keep_observations_aliases_and_chemistry_specific_identity(tmp_path):
    a, b = tmp_path / 'first.csv', tmp_path / 'second.csv'
    write_panel(a, [('x', 'AACCGTATCGTA', 'PS gapmer'), ('y', 'TTGCAACGATCA', 'MOE/PS'),
                    ('alias', 'AACCGTATCGTA', 'PS gapmer')])
    write_panel(b, [('x', 'AACCGTATCGTA', 'MOE/PS'), ('y', 'TTGCAACGATCA', 'MOE/PS'),
                    ('z', 'CATGCATACGAA', 'PS gapmer')])
    context = {'provided_aso_files': [{'path': str(a), 'label': 'Pilot'},
                                     {'path': str(b), 'label': 'Validation'}],
               'provided_asos': [{'name': 'planner.x', 'sequence': 'AACCGTATCGTA', 'chemistry': 'PS gapmer'}]}
    panels = collect(context)
    assert [(p['label'], len(p['rows'])) for p in panels] == [('Pilot', 3), ('Validation', 3)]
    payload = pipeline._multi_set_exact_overlap_payload(panels)
    assert (payload['observation_count'], payload['distinct_sequence_count'], payload['molecule_count']) == (6, 3, 4)
    assert payload['pairwise_overlaps'][0]['shared_count'] == 2
    assert payload['pairwise_overlaps'][0]['shared_molecule_count'] == 1
    molecule = next(m for m in payload['molecule_inventory'] if m['sequence_5to3'] == 'AACCGTATCGTA' and m['recorded_chemistry'] == 'PS gapmer')
    assert [o['candidate_id'] for o in molecule['observations']] == ['x', 'alias']
    table = pipeline._comparison_observation_table_rows(panels)
    assert len(table) == 6
    assert {r['source_artifact'] for r in table} == {str(a), str(b)}
    assert {r['recorded_chemistry'] for r in table} == {'PS gapmer', 'MOE/PS'}


def test_file_labels_do_not_merge_paths_and_explicit_row_sets_take_precedence(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    write_panel(a, [('x', 'AACCGTATCGTA', 'PS', 'actual_set')], with_sets=True)
    write_panel(b, [('x', 'AACCGTATCGTA', 'PS')])
    context = {'provided_aso_files': [{'path': str(a), 'label': 'Same'},
                                     {'path': str(b), 'label': 'Same'}, {'path': str(b)}]}
    panels = collect(context)
    assert len(panels) == 2
    assert panels[0]['set_id'] == 'actual_set'
    assert panels[0]['label'] == 'actual_set'
    assert len({p['set_id'] for p in panels}) == 2
    assert sum(len(p['rows']) for p in panels) == 2


def test_evaluated_rows_restore_file_scope_with_reused_candidate_ids(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    write_panel(a, [('same_id', 'AACCGTATCGTA', 'PS')])
    write_panel(b, [('same_id', 'AACCGTATCGTA', 'MOE/PS')])
    context = {'provided_aso_files': [str(a), str(b)]}
    rows = pipeline._normalize_provided_asos(context)
    for row in rows:
        row['mapping_hits'] = [{'evidence': row['source_file']}]
    panels = collect(context, {'provided_asos': rows})
    assert len(panels) == 2
    assert {p['rows'][0]['mapping_hits'][0]['evidence'] for p in panels} == {str(a), str(b)}
    result = pipeline._multi_set_exact_overlap_payload(panels)
    assert result['pairwise_overlaps'][0]['shared_count'] == 1
    assert result['pairwise_overlaps'][0]['shared_molecule_count'] == 0


def test_unknown_chemistry_does_not_assert_molecular_identity():
    panels = [pipeline._build_compare_set(set_id=key, label=key, source_type='provided',
              rows=[{'name': key, 'sequence': 'AACCGTATCGTA', 'chemistry': chemistry}])
              for key, chemistry in [('a', None), ('b', 'PS')]]
    result = pipeline._multi_set_exact_overlap_payload(panels)
    assert result['pairwise_overlaps'][0]['shared_count'] == 1
    assert result['pairwise_overlaps'][0]['shared_molecule_count'] is None
    assert result['molecule_count'] is None
    assert result['known_molecule_count'] == 1
    assert result['unresolved_molecule_observations'] == 1


def test_molecule_id_column_is_retained_as_a_supplied_alias(tmp_path):
    path = tmp_path / 'aliased.csv'
    path.write_text('molecule_id,sequence_5to3,chemistry\ncustomer_alias,AACCGTATCGTA,PS\n')
    rows = pipeline._normalize_provided_asos({'provided_aso_files': [str(path)]})
    assert rows[0]['name'] == 'customer_alias'
    assert rows[0]['source_row'] == 2


def test_explicit_inline_observation_and_different_chemistry_are_retained(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    write_panel(a, [('a', 'AACCGTATCGTA', 'PS')])
    write_panel(b, [('b', 'TTGCAACGATCA', 'PS')])
    context = {'provided_aso_files': [str(a), str(b)]}
    rows = pipeline._normalize_provided_asos(context) + [
        {'name': 'additional', 'sequence': 'AACCGTATCGTA', 'chemistry': 'PMO'},
        {'name': 'explicit_duplicate', 'sequence': 'AACCGTATCGTA', 'chemistry': 'PS', 'input_origin': 'inline_user'},
    ]
    panels = collect(context, {'provided_asos': rows})
    assert len(panels) == 3
    assert [r['name'] for r in panels[-1]['rows']] == ['additional', 'explicit_duplicate']
