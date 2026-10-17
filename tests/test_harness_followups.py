from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from aso_chemistry import gapmer_architecture, encode_idt_oligo
from aso_agents import pipeline_tools as pipeline
from aso_agents.chemistry_contracts import chemistry_contract_errors
from aso_agents.goal_contract import CANONICAL_TARGET_TYPES
from aso_agents.evaluation_evidence import attach_measurement_references, bind_assessment_measurements
from aso_agents.evaluation_report import write_evaluation_evidence
from aso_agents.tabular_reports import render_report_tables
import run_goal_intake as runtime


@pytest.mark.parametrize("w,g,v", [(2,8,5),(4,10,4),(6,8,3),(3,12,4),(5,10,5),(3,10,3)])
def test_requested_architecture_is_preserved_across_native_boundaries(w,g,v):
    label=f"{w}-{g}-{v}"; length=w+g+v
    payload={"recommendation":{"CHEMISTRY":label,"length":length,"platform":"gapmer","rationale":"MOE wings protect the DNA-gap architecture; activity remains unmeasured."}}
    assert chemistry_contract_errors(payload,length)==[]
    assert runtime._normalize_requested_chemistry(payload)==payload
    assert pipeline._canonicalize_chemistry_payload(payload)==payload
    assert pipeline._apply_requested_length_to_chemistry_payload(payload,length)==payload
    sequence=("ACGT"*((length+3)//4))[:length]
    units=encode_idt_oligo(sequence,label).split("*")
    assert len(units)==length
    assert runtime._apply_idt_configuration(sequence,label)=="*".join(units)
    assert ["2MOEr" in unit for unit in units]==[not w<=i<w+g for i in range(length)]
    recovered="".join(unit[-2] if unit.endswith("/") else unit for unit in units)
    assert recovered==sequence
    assert [runtime._variant_in_chemistry_gap(label,i) for i in range(1,length+1)]==[w<i<=w+g for i in range(1,length+1)]
    with pytest.raises(ValueError):
        runtime._apply_idt_configuration(sequence+"A",label)
    with pytest.raises(ValueError):
        pipeline._apply_requested_length_to_chemistry_payload(payload,length+1)


@pytest.mark.parametrize("label",["gapmer","18-nt 4-10-4 MOE gapmer","0-10-5","4-0-4","3-10","4-10-4-extra","-4-10-4","4.5-10-4","steric blocker", "PMO"])
def test_unsupported_or_descriptive_chemistry_is_not_silently_replaced(label):
    assert runtime._normalize_chemistry_label(label) is None
    goal={"design_programs":[{"program_id":"p","modality":"gapmer","chemistry":label}]}
    runtime._normalize_design_programs_in_goal(goal)
    assert goal["design_programs"][0]["chemistry"]==label
    with pytest.raises(ValueError):
        runtime._resolve_gapmer_chemistry({"chemistry":label},tile_len=20)


def test_standalone_tiler_and_native_encoder_are_identical():
    path=Path(__file__).resolve().parents[1]/"aso-designer-main/aso-designer-main/filter_seqs.py"
    spec=importlib.util.spec_from_file_location("followup_filter",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for label in ("2-8-5","4-10-4","3-12-4","steric-hindrance"):
        architecture=gapmer_architecture(label)
        sequence=("ACGT"*6)[:architecture.length if architecture else 18]
        assert module.apply_configuration(sequence,label)==runtime._apply_idt_configuration(sequence,label)
    with pytest.raises(ValueError):
        module.apply_configuration("ACGT","unsupported")


@pytest.mark.parametrize("kind",["splice_acceptor","splice_donor","splice_junction","splice_site_window"])
def test_reference_feature_types_do_not_require_an_isoform_abundance_goal(kind):
    goal={"intent":"splice_modulation","design_programs":[{
        "program_id":"p","label":"site blockade","modality":"steric","intent":"splice_modulation",
        "objective":"Reduce usage of a selected site","desired_delta":"decrease","transcript_id":"TX.3"}],
        "evaluation_targets":[{"target_type":kind,"objective_type":"route_target","desired_delta":"decrease","transcript_id":"TX.3"}]}
    assert pipeline._canonical_design_goal_contract_issues(goal)==[]
    original=deepcopy(goal)
    goal["evaluation_targets"][0]["target_type"]="junction receiving the upstream intron"
    assert pipeline._canonical_design_goal_contract_issues(goal)==["context.structured_goal.evaluation_targets[0].target_type"]
    schema=pipeline._canonical_issue_expected_schema("context.structured_goal.evaluation_targets[0].target_type")
    assert set(schema["enum"])==CANONICAL_TARGET_TYPES
    assert original["evaluation_targets"][0]["target_type"]==kind


def test_canonical_windows_cannot_be_augmented_by_legacy_prose_inference(monkeypatch):
    goal={"design_programs":[{"program_id":"p","modality":"steric","region_constraints":[{
        "constraint_id":"exact","source_type":"splice_relative_window","aso_overlap_requirement":"full_containment",
        "splice_relative_window":{"anchor_site":"donor","exon_label":4,"relative_start_nt":1,"relative_end_nt":60,"feature_name":"named feature"}}]}]}
    before=deepcopy(goal)
    monkeypatch.setattr(pipeline,"_request_splice_relative_window_programs_from_text",lambda *a:pytest.fail("must not reinterpret an authoritative structured window"))
    assert not pipeline._apply_request_splice_relative_window_programs({"request":"a descriptive request"},goal,{})
    assert goal==before


def test_nested_citations_are_short_and_complete_evidence_is_preserved(tmp_path):
    row={"input_row_id":"neutral","name":"probe","sequence":"AACCGGTT", "mapping_hits":[],
         "near_match_candidates":[{"transcript_id":f"TX{i}","match_orientation":"reverse_complement","mismatch_count":2,"mapping_scope":"spliced_cdna",
                                  "genomic_segments":[{"chrom":"1","start":101,"end":108}]} for i in range(400)]}
    summary=attach_measurement_references({"provided_asos":[row]})
    token=row["reference_tokens"]["evidence_inventory"]
    response={"answer_summary":"Binding is unmeasured; inspect "+token,
              "evaluation_assessments":[{"input_row_id":"neutral","name":"probe","mechanistic_plausibility":"Conditional hypothesis "+token,
                                         "uncertainty":"No experimentally measured activity.","evidence_refs":[token]}]}
    before=deepcopy(response); result=bind_assessment_measurements(response,summary)
    assert len(result["markdown"])<9000
    assert "evaluation_evidence.html#R001_evidence_inventory" in result["markdown"]
    assert response==before and result["original_unbound_response"]==before
    result["response_evaluation_summary"]=summary
    paths=write_evaluation_evidence(tmp_path,result)
    evidence=json.loads(Path(paths["evaluation_evidence_json"]).read_text())
    assert evidence["measurement_citations"]==result["measurement_citations"]
    assert len(evidence["provided_asos"][0]["near_match_candidates"])==400
    assert "No experimentally measured activity." in result["markdown"]


def test_tabular_report_order_uses_recorded_ranks_and_keeps_unknowns(tmp_path):
    csv_path=tmp_path/"observations.csv"; csv_path.write_text("id,rank,score\na,1,0.8\nb,4,0.3\nc,,NA\nd,2,0.7\ne,3,0.5\n")
    report=tmp_path/"sandbox_summary.md"; report.write_text("# Results\n\n{{table:candidates}}\n\nActivity is unmeasured.")
    before=csv_path.read_bytes()
    result={"report_tables":[{"id":"candidates","source_path":str(csv_path),"rank_column":"rank","columns":["id","rank","score"]}]}
    render_report_tables(result,[csv_path],report,required=True)
    body=report.read_text()
    assert [line.split("|")[1].strip() for line in body.splitlines() if line.startswith("| ")][2:]==["a","d","e","b","c"]
    assert "| c |  | NA |" in body and "Activity is unmeasured." in body
    assert csv_path.read_bytes()==before
    assert result["report_table_rendering"][0]["source_sha256"]==hashlib.sha256(before).hexdigest()


def test_declared_export_rejects_independent_row_transcription_with_zero_lost(tmp_path):
    csv_path=tmp_path/"observations.csv"
    csv_path.write_text("candidate_id,exact_hits,mismatch_hits\na,0,\nb,1,3\n")
    report=tmp_path/"report.md"
    report.write_text("# Comparison\n\n| candidate_id | exact_hits | mismatch_hits |\n"
                      "|---|---|---|\n| a | unknown | unknown |\n\n{{table:observations}}\n")
    summary=tmp_path/"sandbox_summary.md"
    summary.write_text("{{table:observations}}\n")
    result={"report_tables":[{"id":"observations","source_path":str(csv_path),
                              "columns":["candidate_id","exact_hits","mismatch_hits"],
                              "rank_column":None}]}
    with pytest.raises(ValueError,match="independently transcribes"):
        render_report_tables(result,[csv_path,report],summary,required=True)
    assert "| a | unknown | unknown |" in report.read_text()
    report.write_text("# Comparison\n\n{{table:observations}}\n")
    render_report_tables(result,[csv_path,report],summary,required=True)
    assert "| a | 0 |  |" in report.read_text()


def test_missing_rank_column_or_unbound_table_cannot_claim_success(tmp_path):
    path=tmp_path/"table.csv";path.write_text("id,value\na,0\n")
    report=tmp_path/"sandbox_summary.md";report.write_text("{{table:values}}")
    with pytest.raises(ValueError,match="Declare report_tables"):
        render_report_tables({},[path],report,required=True)
    with pytest.raises(ValueError,match="rank_column"):
        render_report_tables({"report_tables":[{"id":"values","source_path":str(path),"rank_column":"absent"}]},[path],report,required=True)


def test_region_views_keep_geometry_counts_and_merge_only_identical_populations(tmp_path):
    from aso_agents.binding_region_reports import build_binding_region_report
    rows=[{"Chrom":"chr2","Start":a,"End":a+4,"Sequence":"AACC","Synthesis":"GGTT","Score":1} for a in (100,109)]
    path=tmp_path/"scored.json";path.write_text(json.dumps({"rows":rows}))
    constraints=[{"constraint_id":name,"label":name,"materialized_rows":[["chr2",100,112,name,0,"+"]],"aso_overlap_requirement":rule}
                 for name,rule in (("strict","full_containment"),("strict alias","full_containment"),("overlapping","any_overlap"))]
    result=build_binding_region_report({"materialized_region_constraints":{"global":constraints}},
        {"tiling_results":{"scored_tiles_path":str(path)}},tmp_path,runtime._tile_row_overlap_intervals,{})
    assert len(result["sets"])==2
    assert [group["binding_record_count"] for group in result["sets"]]==[1,2]
    assert [group["distinct_molecule_count"] for group in result["sets"]]==[1,1]
    assert len(result["sets"][0]["source_groups"])==2
