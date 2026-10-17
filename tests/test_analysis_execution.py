import json
import pytest

from aso_agents.analysis_execution import execute_analysis_program, _inexact_decimal_boundary_lines


def test_report_rejects_failure_reasons_for_pass_or_unknown(tmp_path):
    import pytest
    from aso_agents.tabular_reports import render_report_tables

    export = tmp_path / "audit.csv"
    summary = tmp_path / "summary.md"
    summary.write_text("{{table:audit}}")
    for status in ("eligible", "indeterminate"):
        export.write_text("id,eligibility,fail_reasons\nA," + status + ",observed_pass\n")
        with pytest.raises(ValueError, match="nonempty fail_reasons"):
            render_report_tables({"report_tables": [{"id": "audit", "source_path": str(export)}]},
                                 [export], summary, required=True)


def test_report_rejects_counts_and_walk_that_disagree_with_selected_csv(tmp_path):
    from aso_agents.tabular_reports import render_report_tables

    ranked = tmp_path / "ranked.csv"
    selected = tmp_path / "selected.csv"
    report = tmp_path / "report.md"
    summary = tmp_path / "summary.md"
    ranked.write_text("id,set,rank\nX,A,1\nY,A,2\nZ,B,3\n")
    selected.write_text("id,set,rank\nX,A,1\nZ,B,3\n")
    summary.write_text("Analysis complete.")
    report.write_text("| Set | Eligible | Selected |\n|---|---|---|\n"
                      "| A | 1 | 1 |\n| B | 1 | 1 |\n")
    render_report_tables({}, [ranked, selected, report], summary, required=True)
    assert "| A | 2 | 1 |" in report.read_text()

    report.write_text("| Set | Eligible | Selected |\n|---|---|---|\n"
                      "| A | 2 | 1 |\n| B | 1 | 1 |\n\n"
                      "| ID | Set | Accept? |\n|---|---|---|\n"
                      "| X | A | no |\n| Y | A | no |\n| Z | B | yes |\n")
    summary.write_text("Analysis complete.")
    render_report_tables({}, [ranked, selected, report], summary, required=True)
    assert "| X | A | yes |" in report.read_text()
    assert "| Y | A | no |" in report.read_text()


def test_report_can_render_readonly_completed_design_table(tmp_path):
    from aso_agents.tabular_reports import render_report_tables

    panel = tmp_path / "native-panel.csv"
    report = tmp_path / "requested-report.md"
    summary = tmp_path / "summary.md"
    panel.write_text("candidate,sequence,rank\nG1,ACGT,1\nG2,TGCA,2\n")
    report.write_text("Panel\n\n{{table:panel}}\n")
    summary.write_text("{{table:panel}}")
    result = {"report_tables": [{"id": "panel", "source_path": str(panel),
                                 "columns": ["candidate", "sequence", "rank"],
                                 "rank_column": "rank", "preview_limit": None}]}
    render_report_tables(result, [report], summary, required=True, source_tables=[panel])
    assert "| G1 | ACGT | 1 |" in report.read_text()
    assert "| G2 | TGCA | 2 |" in report.read_text()
    assert result["report_table_rendering"][0]["row_count"] == 2


def test_report_replaces_a_second_manual_candidate_table_with_native_rows(tmp_path):
    from aso_agents.tabular_reports import render_report_tables

    panel = tmp_path / "native-panel.csv"
    report = tmp_path / "report.md"
    summary = tmp_path / "summary.md"
    panel.write_text("candidate,antisense_5to3\nG1,ACGT\n")
    report.write_text("| Rank | Antisense |\n|---|---|\n| 1 | |\n\n{{table:panel}}\n")
    summary.write_text("{{table:panel}}")
    result = {"report_tables": [{"id": "panel", "source_path": str(panel)}]}
    render_report_tables(result, [report], summary, required=True, source_tables=[panel])
    assert "| G1 | ACGT |" in report.read_text()
    assert "| 1 | |" not in report.read_text()
    assert result["source_table_canonicalization"]["replaced_manual_tables"] == 1


def test_completed_design_rows_are_exposed_with_their_actual_schema(tmp_path):
    import json
    from aso_agents.pipeline_tools import _build_sandbox_declared_inputs

    child = tmp_path / "child"
    child.mkdir()
    source = child / "generated_aso_set.json"
    panel = child / "design_panel.csv"
    source.write_text(json.dumps({"rows": [{"synthesized_antisense_sequence": "ACGT", "site_label": "cdna60-63", "selection_rank": 1}]}))
    panel.write_text("candidate,antisense_5to3\nG1,ACGT\n")
    (child / "artifacts/intake").mkdir(parents=True)
    (child / "artifacts/intake/structured_goal.json").write_text(json.dumps({"gene_symbol": "GENERIC", "transcript_id": "ENST_GENERIC.1"}))
    (child / "report.md").write_text("The exact-match screen covered the stated reference.\n")
    state = {"task_graph": {"tasks": [{"task_id": "design", "task_type": "design_new_asos"}]},
             "task_artifacts": {"design": {"child_status": "run_complete", "child_run_dir": str(child),
                                            "generated_aso_set_json": str(source), "design_panel_csv": str(panel)}}}
    declared = _build_sandbox_declared_inputs(state, {"tasks": [], "artifact_refs": []})
    candidates = declared["completed_design_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["source_row_count"] == 1
    assert candidates[0]["source_csv_path"] == str(panel)
    assert candidates[0]["rows"][0]["synthesized_antisense_sequence"] == "ACGT"
    assert candidates[0]["reference_context"]["transcript_id"] == "ENST_GENERIC.1"
    assert "exact-match screen" in candidates[0]["native_report"]["text"]


def test_global_splice_panel_coalesces_overlapping_intron_slices():
    from aso_agents.pipeline_tools import _coalesce_global_splice_window_programs

    def program(index, start, end, *, objective="increase inclusion"):
        return {"program_id": f"site{index}", "modality": "steric", "intent": "splice_modulation",
                "objective": objective, "desired_effect": "inclusion", "desired_delta": "increase",
                "site_kind": "donor", "site_context": "downstream_intron", "exon_label": 4,
                "targeting_policy": "explicit_user_region", "length": 20, "candidate_limit": 3,
                "region_constraints": [{"constraint_id": f"w{index}", "source_type": "splice_relative_window",
                                        "operator": "union", "aso_overlap_requirement": "full_containment",
                                        "coordinate_space": "cdna", "exon_label": 4,
                                        "splice_relative_window": {"reference_transcript_id": "ENST_GENERIC.1",
                                                                   "exon_label": 4, "anchor_site": "donor",
                                                                   "relative_start_nt": start, "relative_end_nt": end}}]}

    goal = {"candidate_limit": 3, "min_asos_per_site": 2}
    merged = _coalesce_global_splice_window_programs(
        [program(1, 1, 20), program(2, 11, 30), program(3, 21, 30)], goal)
    assert len(merged) == 1
    assert merged[0]["region_constraints"][0]["splice_relative_window"]["relative_start_nt"] == 1
    assert merged[0]["region_constraints"][0]["splice_relative_window"]["relative_end_nt"] == 30
    assert len(merged[0]["coalesced_window_provenance"]) == 3
    assert "min_asos_per_site" not in goal
    separate = [program(1, 1, 20), program(2, 21, 40, objective="decrease skipping")]
    assert len(_coalesce_global_splice_window_programs(separate, {"candidate_limit": 3})) == 2
    assert len(_coalesce_global_splice_window_programs([program(1, 1, 20), program(2, 11, 30)],
                                                        {"candidate_limit": 3, "candidate_limit_per_site": 3})) == 2


def test_analysis_result_validator_rejects_a_contradicted_completion(tmp_path):
    code = "from pathlib import Path\nPath('sandbox_result.json').write_text('{\"candidate_count\":0}')\nPath('sandbox_summary.md').write_text('No candidates')\n"
    def validate(result):
        if result["candidate_count"] == 0:
            raise ValueError("completed source candidates were not processed")
    outcome = execute_analysis_program(directory=tmp_path, payload={},
                                       generate=lambda _: {"python_code": code},
                                       required_outputs=[], validate_result=validate, max_attempts=1)
    assert not outcome["ok"]
    assert "completed source candidates" in outcome["error"]


def test_completed_physical_evaluation_rejects_header_only_export_and_repairs(tmp_path):
    export = tmp_path / "delivered.csv"
    requests = []
    def generate(request):
        requests.append(request)
        if len(requests) == 2:
            assert "zero data rows" in request["execution_repair"]["error"]
            assert "2 physical molecule rows" in request["execution_repair"]["error"]
        content = "aso_id,sequence\n" + ("A,ACGT\nB,TGCA\n" if len(requests) == 2 else "")
        return {"python_code": (
            "from pathlib import Path\n"
            "Path('sandbox_result.json').write_text('{}')\n"
            "Path('sandbox_summary.md').write_text('Evaluation exported.')\n"
            f"Path({str(export)!r}).write_text({content!r})\n"
        )}
    payload = {"operation": "render_requested_outputs", "declared_inputs": {
        "completed_run": {"evaluation_evidence": [{"task_id": "evaluated",
            "total_rows": 2, "rows": [{"name": "A"}, {"name": "B"}]}]}}}
    outcome = execute_analysis_program(directory=tmp_path / "analysis", payload=payload,
                                       generate=generate, required_outputs=[export], max_attempts=2)
    assert outcome["ok"] and len(outcome["attempts"]) == 2
    assert export.read_text() == "aso_id,sequence\nA,ACGT\nB,TGCA\n"


def test_header_only_export_is_valid_when_native_evaluation_has_no_rows(tmp_path):
    export = tmp_path / "delivered.csv"
    code = ("from pathlib import Path\n"
            "Path('sandbox_result.json').write_text('{}')\n"
            "Path('sandbox_summary.md').write_text('No records to export.')\n"
            f"Path({str(export)!r}).write_text('aso_id,sequence\\n')\n")
    payload = {"operation": "render_requested_outputs", "declared_inputs": {
        "completed_run": {"evaluation_evidence": [{"task_id": "evaluated", "total_rows": 0,
            "rows": []}]}}}
    outcome = execute_analysis_program(directory=tmp_path / "analysis", payload=payload,
                                       generate=lambda _: {"python_code": code},
                                       required_outputs=[export], max_attempts=1)
    assert outcome["ok"]


@pytest.mark.parametrize("mutation", ["unlink", "directory", "symlink"])
def test_protected_output_restored_after_destructive_generated_operation(tmp_path, mutation):
    protected = tmp_path / "delivered.csv"
    protected.write_text("id,value\n001,0\n")
    original = protected.read_bytes()
    replacement = tmp_path / "other.csv"
    replacement.write_text("Other producer's observations")
    code = "from pathlib import Path\nPath('sandbox_result.json').write_text('{}')\nPath('sandbox_summary.md').write_text('Complete')\n"
    code += f"p=Path({str(protected)!r})\np.unlink()\n"
    if mutation == "directory":
        code += "p.mkdir()\n"
    elif mutation == "symlink":
        code += f"p.symlink_to({str(replacement)!r})\n"
    result = execute_analysis_program(directory=tmp_path, payload={}, generate=lambda _: {"python_code": code},
                                      required_outputs=[], readonly_outputs=[protected], max_attempts=1)
    assert result["ok"] is False and "original contents restored" in result["error"]
    assert protected.read_bytes() == original and not protected.is_symlink()
    assert replacement.read_text() == "Other producer's observations"


def test_execution_contract_and_misplaced_outputs_reach_repair_without_accepting_them(tmp_path):
    import json
    export = tmp_path / 'requested.csv'
    calls = []
    def generate(request):
        calls.append(request)
        contract = request['execution_contract']
        assert contract['working_directory'] == str(tmp_path)
        assert contract['required_output_files'] == [str(tmp_path / 'sandbox_result.json'), str(tmp_path / 'sandbox_summary.md'), str(export)]
        if len(calls) == 1:
            return {'python_code': "from pathlib import Path\nPath('wrong.csv').write_text('value\\n7\\n')"}
        previous = request['execution_repair']['previous_attempts'][0]
        assert 'wrong.csv' in previous['workspace_files']
        assert previous['error'] and not export.exists()
        return {'python_code': 'from pathlib import Path\n' + '\n'.join(
            f'Path({str(path)!r}).write_text({content!r})' for path, content in [
                (contract['result_json'], json.dumps({'summary': 'One recorded value'})),
                (contract['summary_markdown'], 'One recorded value: 7.'), (export, 'value\n7\n')])}
    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate, required_outputs=[export])
    assert result['ok'] and len(result['attempts']) == 2


def test_fresh_same_named_working_exports_reach_declared_paths(tmp_path):
    work = tmp_path / 'work'
    requested = [tmp_path / 'delivered' / 'audit.csv', tmp_path / 'delivered' / 'audit.md']
    code = ("from pathlib import Path\n"
            "Path('sandbox_result.json').write_text('{}')\n"
            "Path('sandbox_summary.md').write_text('Eight supplied rows audited.')\n"
            "Path('audit.csv').write_text('id,status\\nA,eligible\\n')\n"
            "Path('audit.md').write_text('A is eligible.')\n")
    result = execute_analysis_program(directory=work, payload={},
                                      generate=lambda _: {'python_code': code},
                                      required_outputs=requested, max_attempts=1)
    assert result['ok']
    assert requested[0].read_text() == 'id,status\nA,eligible\n'
    assert requested[1].read_text() == 'A is eligible.'
    assert len(result['attempts'][0]['promoted_same_named_exports']) == 2


def test_fresh_output_subdirectory_export_reaches_declared_path(tmp_path):
    work = tmp_path / 'work'
    requested = tmp_path / 'delivered' / 'audit.csv'
    code = ("from pathlib import Path\n"
            "Path('sandbox_result.json').write_text('{}')\n"
            "Path('sandbox_summary.md').write_text('Three rows assessed.')\n"
            "Path('output').mkdir()\n"
            "Path('output/audit.csv').write_text('id,status\\nA,eligible\\n')\n")
    result = execute_analysis_program(directory=work, payload={},
                                      generate=lambda _: {'python_code': code},
                                      required_outputs=[requested], max_attempts=1)
    assert result['ok']
    assert requested.read_text() == 'id,status\nA,eligible\n'
    assert result['attempts'][0]['promoted_same_named_exports'] == [
        {'source': str(work / 'output' / 'audit.csv'), 'requested_output': str(requested)}]


def test_ambiguous_root_and_output_subdirectory_exports_are_not_delivered(tmp_path):
    work = tmp_path / 'work'
    requested = tmp_path / 'delivered' / 'audit.csv'
    code = ("from pathlib import Path\n"
            "Path('sandbox_result.json').write_text('{}')\n"
            "Path('sandbox_summary.md').write_text('Two files conflict.')\n"
            "Path('output').mkdir()\n"
            "Path('audit.csv').write_text('id,value\\nA,1\\n')\n"
            "Path('output/audit.csv').write_text('id,value\\nA,2\\n')\n")
    result = execute_analysis_program(directory=work, payload={},
                                      generate=lambda _: {'python_code': code},
                                      required_outputs=[requested], max_attempts=1)
    assert not result['ok'] and 'not created' in result['error']
    assert not requested.exists()


def test_stale_output_subdirectory_export_is_not_delivered(tmp_path):
    work = tmp_path / 'work'
    (work / 'output').mkdir(parents=True)
    (work / 'output' / 'audit.csv').write_text('old,wrong\n')
    requested = tmp_path / 'delivered' / 'audit.csv'
    code = ("from pathlib import Path\n"
            "Path('sandbox_result.json').write_text('{}')\n"
            "Path('sandbox_summary.md').write_text('New summary only.')\n")
    result = execute_analysis_program(directory=work, payload={},
                                      generate=lambda _: {'python_code': code},
                                      required_outputs=[requested], max_attempts=1)
    assert not result['ok'] and 'not created' in result['error']
    assert not requested.exists()


def test_output_subdirectory_symlink_is_not_promoted(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    external = tmp_path / 'external'
    external.mkdir()
    (work / 'output').symlink_to(external, target_is_directory=True)
    requested = tmp_path / 'delivered' / 'audit.csv'
    code = ("from pathlib import Path\n"
            "Path('sandbox_result.json').write_text('{}')\n"
            "Path('sandbox_summary.md').write_text('Alias must not be followed.')\n"
            "Path('output/audit.csv').write_text('id,value\\nA,1\\n')\n")
    result = execute_analysis_program(directory=work, payload={},
                                      generate=lambda _: {'python_code': code},
                                      required_outputs=[requested], max_attempts=1)
    assert not result['ok'] and 'not created' in result['error']
    assert not requested.exists()


def test_stale_same_named_working_export_is_not_delivered(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'audit.csv').write_text('old,wrong\n')
    requested = tmp_path / 'delivered' / 'audit.csv'
    code = ("from pathlib import Path\n"
            "Path('sandbox_result.json').write_text('{}')\n"
            "Path('sandbox_summary.md').write_text('New summary only.')\n")
    result = execute_analysis_program(directory=work, payload={},
                                      generate=lambda _: {'python_code': code},
                                      required_outputs=[requested], max_attempts=1)
    assert not result['ok'] and 'not created' in result['error']
    assert not requested.exists()


def test_missing_export_repairs_program_with_original_inputs(tmp_path):
    export = tmp_path / 'requested.csv'
    original = {'declared_inputs': {'rows': [{'id':'r1','value':0}, {'id':'r2','value':None}]}}
    seen = []
    def generate(payload):
        seen.append(payload)
        program = "from pathlib import Path\nPath('sandbox_result.json').write_text('{}')\nPath('sandbox_summary.md').write_text('Two supplied rows were analyzed.')\n"
        if len(seen) == 2:
            program += f"Path({str(export)!r}).write_text('id,value\\nr1,0\\nr2,NA\\n')\n"
        return {'python_code':program}
    result = execute_analysis_program(directory=tmp_path, payload=original, generate=generate, required_outputs=[export])
    assert result['ok'] and len(result['attempts']) == 2
    assert 'not created' in seen[1]['execution_repair']['error']
    assert seen[1]['declared_inputs'] == original['declared_inputs']
    assert export.read_text() == 'id,value\nr1,0\nr2,NA\n'
    assert json.loads((tmp_path/'execution_attempts/01/execution.json').read_text())['error']


def test_repeated_exception_never_becomes_a_successful_fallback_report(tmp_path):
    result = execute_analysis_program(directory=tmp_path, payload={},
        generate=lambda _: {'python_code':"raise ValueError('bad column name')"}, required_outputs=[])
    assert not result['ok'] and len(result['attempts']) == 3
    assert 'bad column name' in result['error']
    assert not (tmp_path/'sandbox_summary.md').exists()


def test_inexact_decimal_boundary_is_rejected_before_execution_and_repaired(tmp_path):
    calls = []
    def generate(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {'python_code':
                "from pathlib import Path\n"
                "gc = 10 / 20\nrep = float('0.47')\n"
                "Path('unsafe_side_effect').write_text('executed')\n"
                "disagreement = abs(gc - rep) > 0.03\n"}
        assert 'Inexact decimal-boundary comparison' in payload['execution_repair']['error']
        return {'python_code':
            "from fractions import Fraction\nfrom decimal import Decimal\n"
            "from pathlib import Path\n"
            "gc = Fraction(10, 20)\nrep = Decimal('0.47')\n"
            "disagreement = abs(Decimal(gc.numerator) / Decimal(gc.denominator) - rep) > Decimal('0.03')\n"
            "Path('sandbox_result.json').write_text('{' + '\"disagreement\": false' + '}')\n"
            "Path('sandbox_summary.md').write_text(str(disagreement))\n"}
    result = execute_analysis_program(directory=tmp_path, payload={},
                                      generate=generate, required_outputs=[])
    assert result['ok'] and len(result['attempts']) == 2
    assert not (tmp_path/'unsafe_side_effect').exists()
    assert result['summary'] == 'False'


def test_numeric_guard_keeps_safe_and_unrelated_float_calculations():
    safe = "from decimal import Decimal\nx = Decimal('0.50')\ny = Decimal('0.47')\nflag = abs(x-y) > Decimal('0.03')\n"
    unrelated = "import math\nrate = float('0.47')\nscore = math.exp(rate)\nflag = score > 1\n"
    assert _inexact_decimal_boundary_lines(safe) == []
    assert _inexact_decimal_boundary_lines(unrelated) == []
    assert _inexact_decimal_boundary_lines("ratio = 10/20\nflag = ratio >= 0.35\n") == [2]


def test_manual_table_missing_marker_is_verified_without_masking_changed_values():
    from aso_agents.tabular_reports import _repeated_export_table_columns
    fields = ['id', 'eligible', 'failure_reasons']
    rows = [{'id':'a', 'eligible':'Y', 'failure_reasons':''},
            {'id':'b', 'eligible':'N', 'failure_reasons':'length'}]
    faithful = '| id | eligible | failure_reasons |\n|---|---|---|\n| a | Y | - |\n| b | N | length |'
    wrong = faithful.replace('| b | N | length |', '| b | Y | length |')
    assert _repeated_export_table_columns(faithful, fields, rows) == []
    assert _repeated_export_table_columns(wrong, fields, rows) == sorted(fields)


def test_report_table_relative_source_is_resolved_from_analysis_directory(tmp_path):
    from aso_agents.tabular_reports import render_report_tables
    output = tmp_path/'audit.csv'
    summary = tmp_path/'summary.md'
    output.write_text('id,value\na,0\n')
    summary.write_text('{{table:audit}}')
    result = {'report_tables':[{'id':'audit','source_path':'audit.csv',
                                'columns':['id','value'],'rank_column':None,
                                'preview_limit':None}]}
    render_report_tables(result,[output],summary,required=True)
    assert result['report_tables'][0]['source_path'] == str(output)
    assert '| a | 0 |' in summary.read_text()


def test_stale_files_do_not_satisfy_current_execution(tmp_path):
    (tmp_path/'sandbox_result.json').write_text('{}')
    (tmp_path/'sandbox_summary.md').write_text('An old result')
    result = execute_analysis_program(directory=tmp_path, payload={},
        generate=lambda _: {'python_code':"print('I created a report')"}, required_outputs=[], max_attempts=1)
    assert not result['ok'] and 'did not produce' in result['error']


def test_execution_repair_receives_real_runtime_types_and_failed_history(tmp_path):
    seen = []
    def generate(payload):
        seen.append(payload)
        return {'analysis_plan': 'Retain supplied values.', 'python_code': "value = 'unavailable'\nprint(f'{value:.3f}')"}
    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[], max_attempts=2)
    assert not result['ok']
    repair = seen[1]['execution_repair']
    assert repair['previous_plan']['analysis_plan'] == 'Retain supplied values.'
    frame = repair['runtime_diagnostics']['frames'][-1]
    assert frame['line'] == 2 and frame['local_types']['value']['type'] == 'builtins.str'
    assert 'unavailable' not in json.dumps(frame)
    assert len(repair['previous_attempts']) == 1


def test_diagnostic_wrapper_preserves_script_arguments_and_task_local_imports(tmp_path):
    (tmp_path / 'local_measurements.py').write_text('VALUE = 0\n')
    code = "import sys, json\nfrom pathlib import Path\nfrom local_measurements import VALUE\n" \
           "assert sys.argv == [str(Path(__file__).resolve())]\n" \
           "Path('sandbox_result.json').write_text(json.dumps({'value': VALUE}))\n" \
           "Path('sandbox_summary.md').write_text('Recorded value is zero.')\n"
    result = execute_analysis_program(directory=tmp_path, payload={},
        generate=lambda _: {'python_code': code}, required_outputs=[], max_attempts=1)
    assert result['ok'] and result['result']['value'] == 0


def test_native_renderer_binds_completed_exports_without_another_code_generation(tmp_path):
    export = tmp_path / 'requested.csv'
    calls = []
    def generate(payload):
        calls.append(payload)
        return {'python_code': "from pathlib import Path\nPath('sandbox_result.json').write_text('{}')\n"
            "Path('sandbox_summary.md').write_text('Missing values remain unknown.')\n"
            f"Path({str(export)!r}).write_text('id,value\\na,0\\nb,NA\\n')\n"}
    result = execute_analysis_program(directory=tmp_path, payload={'require_tabular_report_views': True},
        generate=generate, required_outputs=[export])
    assert result['ok'] and len(calls) == 1
    assert '| a | 0 |' in result['summary'] and '| b | NA |' in result['summary']
    assert result['result']['report_table_rendering'][0]['rank_column'] is None
    assert export.read_text() == 'id,value\na,0\nb,NA\n'


def test_renderer_uses_explicit_declarations_inside_the_result_envelope(tmp_path):
    from aso_agents.tabular_reports import render_report_tables
    export, summary = tmp_path / 'measurements.tsv', tmp_path / 'summary.md'
    export.write_text('sample\tvalue\na\t0\nb\tNA\n')
    summary.write_text('Recorded observations:\n{{table:observations}}')
    declared = {'id': 'observations', 'source_path': str(export), 'columns': ['sample', 'value'],
                'rank_column': None, 'preview_limit': None}
    result = {'analysis_plan': 'Preserve measurements', 'sandbox_result': {'report_tables': [declared]}}
    render_report_tables(result, [export], summary, required=True)
    assert '| a | 0 |' in summary.read_text() and '| b | NA |' in summary.read_text()
    assert result['report_table_declarations_path'] == 'sandbox_result.report_tables'
    assert result['sandbox_result']['report_tables'] == result['report_tables'] == [declared]


def test_renderer_binds_exact_export_filename_marker_without_model_declaration(tmp_path):
    from aso_agents.tabular_reports import render_report_tables
    export, report = tmp_path / 'pilot_comparison.csv', tmp_path / 'pilot_comparison.md'
    summary = tmp_path / 'sandbox_summary.md'
    export.write_text('candidate_id,exact_hits,mismatch_hits\nA2,0,\nB2,0,2\n')
    report.write_text('Recorded rows:\n\n{{table:pilot_comparison.csv}}\n')
    summary.write_text('Recorded rows: {{table:pilot_comparison.csv}}')
    result = {}
    render_report_tables(result, [export, report], summary, required=True)
    assert '| A2 | 0 |  |' in report.read_text()
    assert '| B2 | 0 | 2 |' in summary.read_text()
    assert result['report_tables'][0]['source_path'] == str(export)
    assert result['report_tables'][0]['rank_column'] is None


def test_renderer_keeps_single_named_report_complete_when_marker_is_only_in_summary(tmp_path):
    from aso_agents.tabular_reports import render_report_tables
    export, report = tmp_path / 'pilot_audit.csv', tmp_path / 'pilot_report.md'
    summary = tmp_path / 'sandbox_summary.md'
    export.write_text('candidate_id,sequence,chemistry,status\nA1,ACGT,full-PS,eligible\n')
    report.write_text('# Pilot report\n\n| candidate_id | status |\n|---|---|\n| A1 | eligible |\n')
    summary.write_text('Audit table:\n\n{{table:audit}}\n')
    result = {'report_tables': [{'id': 'audit', 'source_path': str(export),
                                'columns': ['candidate_id', 'sequence', 'chemistry', 'status'],
                                'preview_limit': None}]}
    render_report_tables(result, [export, report], summary, required=True)
    assert '| A1 | ACGT | full-PS | eligible |' in report.read_text()
    assert 'Complete exported source table' in report.read_text()


def test_renderer_allows_a_faithful_subset_table_but_rejects_one_changed_value(tmp_path):
    from aso_agents.tabular_reports import render_report_tables
    export, report = tmp_path / 'candidates.csv', tmp_path / 'comparison.md'
    summary = tmp_path / 'sandbox_summary.md'
    export.write_text('candidate_id,set_id,exact_hits,mismatch_hits\nA1,A,0,\nB1,B,1,2\n')
    report.write_text('| candidate_id | set_id | exact_hits | mismatch_hits |\n'
                      '|---|---|---|---|\n| B1 | B | 1 | 2 |\n'
                      '{{table:candidates.csv}}')
    summary.write_text('{{table:candidates.csv}}')
    render_report_tables({}, [export, report], summary, required=True)
    assert '| A1 | A | 0 |  |' in report.read_text()
    report.write_text('| candidate_id | set_id | exact_hits | mismatch_hits |\n'
                      '|---|---|---|---|\n| A1 | A | unknown |  |\n'
                      '{{table:candidates.csv}}')
    summary.write_text('{{table:candidates.csv}}')
    with pytest.raises(ValueError, match='cannot be verified'):
        render_report_tables({}, [export, report], summary, required=True)


def test_renderer_does_not_guess_arbitrary_or_duplicate_export_markers(tmp_path):
    from aso_agents.tabular_reports import render_report_tables
    export = tmp_path / 'observations.csv'
    summary = tmp_path / 'sandbox_summary.md'
    export.write_text('id,value\na,0\n')
    summary.write_text('{{table:unspecified}}')
    with pytest.raises(ValueError, match='ambiguous'):
        render_report_tables({}, [export], summary, required=True)
    other = tmp_path / 'nested' / 'observations.csv'
    other.parent.mkdir()
    other.write_text('id,value\nb,1\n')
    summary.write_text('{{table:observations.csv}}')
    with pytest.raises(ValueError, match='ambiguous'):
        render_report_tables({}, [export, other], summary, required=True)


def test_renderer_binds_identical_staged_table_to_requested_export(tmp_path):
    from aso_agents.tabular_reports import render_report_tables
    export = tmp_path / 'requested' / 'audit.csv'
    staged = tmp_path / 'sandbox' / 'output' / 'audit.csv'
    summary = tmp_path / 'sandbox' / 'sandbox_summary.md'
    export.parent.mkdir()
    staged.parent.mkdir(parents=True)
    export.write_text('id,value\nr1,0\nr2,NA\n')
    staged.write_bytes(export.read_bytes())
    summary.write_text('Audit results: {{table:audit}}')
    result = {'report_tables': [{'id': 'audit', 'source_path': str(staged),
                                 'columns': ['id', 'value']}]}
    render_report_tables(result, [export], summary, required=True)
    assert '| r1 | 0 |' in summary.read_text()
    assert len(result['report_tables']) == 1
    assert result['report_tables'][0]['source_path'] == str(export)
    assert result['report_table_rendering'][0]['declared_source_path'] == str(staged)
    assert export.read_text() == 'id,value\nr1,0\nr2,NA\n'


def test_renderer_rejects_staging_copy_with_different_data(tmp_path):
    from aso_agents.tabular_reports import render_report_tables
    export = tmp_path / 'requested' / 'audit.csv'
    staged = tmp_path / 'sandbox' / 'output' / 'audit.csv'
    summary = tmp_path / 'sandbox' / 'sandbox_summary.md'
    export.parent.mkdir()
    staged.parent.mkdir(parents=True)
    export.write_text('id,value\nr1,0\n')
    staged.write_text('id,value\nr1,1\n')
    summary.write_text('{{table:audit}}')
    result = {'report_tables': [{'id': 'audit', 'source_path': str(staged)}]}
    with pytest.raises(ValueError, match='actual requested CSV/TSV output'):
        render_report_tables(result, [export], summary, required=True)
    assert export.read_text() == 'id,value\nr1,0\n'


def test_renderer_rejects_conflicting_envelope_declarations_and_unknown_markers(tmp_path):
    import pytest
    from aso_agents.tabular_reports import render_report_tables
    export, summary = tmp_path / 'observations.csv', tmp_path / 'summary.md'
    export.write_text('sample,value\na,0\n')
    summary.write_text('{{table:measurements}}')
    with pytest.raises(ValueError, match='Conflicting'):
        render_report_tables({'report_tables': [{'id': 'a'}],
            'sandbox_result': {'report_tables': [{'id': 'b'}]}}, [export], summary, required=True)
    with pytest.raises(ValueError, match='ambiguous'):
        render_report_tables({}, [export], summary, required=True)
    assert summary.read_text() == '{{table:measurements}}'


def test_comparison_preserves_arbitrary_supplied_metadata(monkeypatch):
    from aso_agents import pipeline_tools as tools
    supplied = {'name':'neutral-1','sequence':'ACGTACGT','length':8,'gc_fraction':0.5,
                'input_metadata':{'batch':'A','custom_metric':'0','other_metric':'NA'}}
    row = tools._compact_provided_aso_for_ranking(supplied, [], None)
    assert row['input_metadata'] == supplied['input_metadata']
    row['input_metadata']['custom_metric'] = 'changed'
    assert supplied['input_metadata']['custom_metric'] == '0'
    captured = []
    monkeypatch.setattr(tools, '_normalize_provided_asos', lambda _: [supplied])
    monkeypatch.setattr(tools, '_run_workbench_response_sync', lambda payload: captured.append(payload) or {'markdown':'A comparison'})
    tools._build_compare_aso_sets_response(state={'request':'Compare my supplied table'}, request_envelope={},
        comparison_payload={}, evaluated_payload={}, generated_rows=[])
    assert captured[0]['provided_aso_evaluation']['provided_asos'][0]['input_metadata'] == supplied['input_metadata']


def test_knockdown_by_exon_skipping_gets_context_without_affecting_plain_gapmers(monkeypatch):
    import run_goal_intake as cli
    monkeypatch.setattr(cli, '_resolve_run_splice_eval', lambda goal: any(p.get('run_splice_eval') for p in goal.get('design_programs', [])))
    goal = {'gene_symbol':'NEUTRAL', 'intent':'knockdown', 'exon_label':3, 'desired_effect':'skipping',
            'splice_isoform_objectives':[{'desired_effect':'skipping','exon_label':3,'objective_scope':'default_exon_context','role':'pathogenic'}],
            'design_programs':[{'modality':'steric','run_splice_eval':True}]}
    assert cli._should_seed_default_splice_context(goal)
    assert not cli._should_seed_default_splice_context({'gene_symbol':'NEUTRAL','intent':'knockdown','design_programs':[{'modality':'gapmer'}]})
