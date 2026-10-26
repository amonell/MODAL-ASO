"""Independent contracts for model inputs, conditional outputs and report delivery."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from aso_chemistry import encode_idt_oligo
from aso_molecule import molecule_record
from aso_agents.oligoai import (MODEL_SHA256, SOURCE_REVISION, evaluate_candidates,
    evaluation_conditions, matching_contexts, model_chemistry, validate_model_input,
    reverse_complement, markdown_section, html_section, target_references,
    append_report_evaluations)
from aso_agents.response_artifacts import digest


SEQ = "ACGTCAGACTTAGCACTGCA"


def row(sequence=SEQ, architecture="5-10-5"):
    notation = encode_idt_oligo(sequence, architecture)
    record = molecule_record(sequence, architecture, notation, assumption_status="configured_intent_unknown")
    return {"sequence": reverse_complement(sequence), "synthesized_antisense_sequence": sequence,
            "chemistry_idt": notation, "source_modality": "gapmer", "selection_rank": 2,
            "molecule_specification": record}


def reference(sequence=SEQ):
    return {"label": "target", "transcript_id": "transcript", "species": "human",
            "kind": "pre_mrna", "sequence": "G"*50 + reverse_complement(sequence) + "C"*50,
            "genomic_start0": 100, "genomic_end0": 220, "strand": "+", "chrom": "chrX",
            "site_start0": 50, "site_end0": 70, "provenance": {"source": "fixture"}}


def predictor(inputs, directory):
    for item in inputs:
        validate_model_input(item)
    return {"model_sha256": MODEL_SHA256, "source_revision": SOURCE_REVISION,
            "predictions": [{"input_id": i["input_id"], "predicted_inhibition_percent": 73.125} for i in inputs]}


def test_physical_identity_and_every_bond_are_preserved(tmp_path):
    rows = [row()]
    before = deepcopy(rows[0])
    result = evaluate_candidates(rows, {}, tmp_path, references=[reference()], predictor=predictor)
    assert result["status"] == "complete"
    assert {k:v for k,v in rows[0].items() if k != "oligoai_evaluation"} == before
    record,sugars,bonds = model_chemistry(rows[0], {})
    assert record == before["molecule_specification"]
    assert sugars == ["MOE"]*5 + ["DNA"]*10 + ["MOE"]*5
    assert bonds == ["PS"]*19
    assert result["conditions"]["dose_basis"] == "assumed_reference_scenario"
    assert result["candidates"][0]["contexts"][0]["predicted_inhibition_percent"] == 73.125
    records = list(csv.DictReader(Path(result["artifact_csv"]).open()))
    assert records[0]["antisense_5to3"] == SEQ
    assert float(records[0]["predicted_inhibition_percent"]) == 73.125


def test_missing_optional_runtime_is_unavailable_without_changing_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("ASO_OLIGOAI_ROOT", str(tmp_path / "not mounted"))
    candidate = row()
    before = deepcopy(candidate)
    result = evaluate_candidates([candidate], {}, tmp_path / "reports", references=[reference()])
    assert result["predicted_count"] == 0
    assert result["candidates"][0]["status"] == "unavailable"
    assert result["runtime_availability"]["status"] == "unavailable"
    assert "not mounted" in result["candidates"][0]["reason"]
    assert "inference_error" not in result
    assert {k: v for k, v in candidate.items() if k != "oligoai_evaluation"} == before


def test_real_predictor_error_is_not_hidden_as_missing_optional_install(tmp_path):
    def broken(*_):
        raise RuntimeError("predictor crashed")
    result = evaluate_candidates([row()], {}, tmp_path, references=[reference()], predictor=broken)
    assert result["candidates"][0]["status"] == "error"
    assert "predictor crashed" in result["inference_error"]
    assert "runtime_availability" not in result


def test_unsupported_ome_is_not_substituted(tmp_path):
    rows = [row(architecture="2ome:5-10-5")]
    result = evaluate_candidates(rows, {}, tmp_path, references=[reference()], predictor=lambda *a: pytest.fail("must not call model"))
    assert result["predicted_count"] == 0
    assert result["candidates"][0]["status"] == "unsupported"
    assert "2OME" in result["candidates"][0]["reason"]
    assert rows[0]["molecule_specification"]["nucleotide_sugars_5to3"][0] == "2OME"


@pytest.mark.parametrize("architecture", ["steric-hindrance", "steric-pmo"])
def test_steric_profiles_do_not_receive_gapmer_predictions(tmp_path, architecture):
    r = row(architecture=architecture) if architecture != "steric-pmo" else {
        "molecule_specification": molecule_record(SEQ, architecture, None), "source_modality": "steric"}
    assert evaluate_candidates([r], {}, tmp_path, predictor=lambda *a: pytest.fail("not gapmer")) == {}


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "100", True])
def test_invalid_dose_never_reaches_model(tmp_path, value):
    result = evaluate_candidates([row()], {"oligoai_evaluation":{"dosage_nm":value}}, tmp_path,
                                 references=[reference()], predictor=lambda *a: pytest.fail("invalid dose"))
    assert result["candidates"][0]["status"] == "unavailable"


def test_explicit_conditions_and_disable(tmp_path):
    options = {"dosage_nm":25, "transfection_method":"Gymnosis"}
    result = evaluate_candidates([row()], {"constraints":{"oligoai_evaluation":options}}, tmp_path,
                                 references=[reference()], predictor=predictor)
    assert result["conditions"]["dosage_nm"] == 25
    assert result["conditions"]["method_basis"] == "explicit_structured_input"
    assert "25 nM" in html_section(result)
    disabled = evaluate_candidates([row()], {"oligoai_evaluation":{"enabled":False}}, tmp_path,
                                   predictor=lambda *a: pytest.fail("disabled"))
    assert disabled["candidates"][0]["status"] == "disabled"


@pytest.mark.parametrize("options", [{"transfection_method":"unknown"}, {"enabled":"yes"}, {"dose":100}])
def test_conditions_reject_unknown_or_misplaced_fields(options):
    with pytest.raises(ValueError):
        evaluation_conditions({"oligoai_evaluation":options})


def test_same_strand_string_is_not_binding_evidence(tmp_path):
    ref = reference()
    ref["sequence"] = "G"*50 + SEQ + "C"*50
    result = evaluate_candidates([row()], {}, tmp_path, references=[ref], predictor=lambda *a: pytest.fail("wrong orientation"))
    assert result["predicted_count"] == 0
    assert "no_exact_antisense" in result["candidates"][0]["reason"]


def test_repeats_boundaries_and_mature_context_label():
    r = reference()
    r.update(kind="mature_rna", sequence=reverse_complement(SEQ)*2, site_start0=0, site_end0=40)
    contexts = matching_contexts(SEQ, [r])
    assert len(contexts) == 2
    assert all(c["rna_context"][50:70].replace("U","T") == reverse_complement(SEQ) for c in contexts)
    assert all(len(c["rna_context"]) == 120 for c in contexts)
    assert contexts[0]["context_basis"] == "mature_rna_extrapolation"
    assert all(c["flank_padding_nt"] == 80 for c in contexts)


def test_native_premrna_is_preferred_over_mature_rna():
    pre = reference()
    mature = {**pre,"kind":"mature_rna","sequence":"A"*50+reverse_complement(SEQ)+"T"*50}
    assert len(matching_contexts(SEQ,[mature,pre])) == 1
    assert matching_contexts(SEQ,[mature,pre])[0]["kind"] == "pre_mrna"


@pytest.mark.parametrize("strand", ["+", "-"])
def test_reference_extraction_respects_rna_orientation(strand):
    full = "G"*50 + reverse_complement(SEQ) + "C"*50
    genome_seq = full if strand == "+" else reverse_complement(full)
    catalog = {"target":{"chrom":"chrX", "strand":strand, "exons_tx":[{"start":51,"end":70}],
                         "reference_binding":{"species":"human","assembly":"test"}}}
    class Contig:
        def __len__(self):return len(genome_seq)
        def __getitem__(self,s):return genome_seq[s]
    runtime = SimpleNamespace(_resolve_patient_reference_fasta=lambda goal:None)
    refs = target_references({},runtime,catalog=catalog,cdna={},genomes={"target":{"chrX":Contig()}})
    contexts = matching_contexts(SEQ,refs)
    assert len(contexts) == 1
    assert contexts[0]["binding_interval_bed0_half_open"] == ["chrX",50,70]
    assert contexts[0]["rna_context"].replace("U","T") == full


@pytest.mark.parametrize("failure", ["exception","nan","missing","duplicate","wrong"])
def test_bad_inference_is_not_fabricated_as_success(tmp_path,failure):
    def bad(inputs,directory):
        if failure=="exception":raise RuntimeError("gpu execution failed")
        result=predictor(inputs,directory)
        if failure=="nan":result["predictions"][0]["predicted_inhibition_percent"]=float("nan")
        if failure=="missing":result["predictions"]=[]
        if failure=="duplicate":result["predictions"]*=2
        if failure=="wrong":result["predictions"][0]["input_id"]="other"
        return result
    result=evaluate_candidates([row()],{},tmp_path,references=[reference()],predictor=bad)
    assert result["predicted_count"]==0
    assert result["candidates"][0]["status"]=="error"
    assert "—" in markdown_section(result)


def test_out_of_range_prediction_is_kept_raw(tmp_path):
    def raw(inputs,directory):
        r=predictor(inputs,directory);r["predictions"][0]["predicted_inhibition_percent"]=110.5;return r
    result=evaluate_candidates([row()],{},tmp_path,references=[reference()],predictor=raw)
    assert "110.50" in markdown_section(result)
    assert "110.50" in html_section(result)


def test_reports_show_same_scores_and_scope_once(tmp_path):
    result=evaluate_candidates([row()],{},tmp_path,references=[reference()],predictor=predictor)
    md=markdown_section(result);ht=html_section(result)
    for text in ("73.12", SEQ, "assumed reference scenario", "Mature-RNA", "Predictions are raw"):
        assert text in md and text in ht
    payload={"evaluation_summary":{"oligoai_evaluation":result},"components":[{"oligoai_evaluation":result}]}
    combined=append_report_evaluations("# Report",payload)
    assert combined.count("## OligoAI gapmer efficacy prediction")==1
    assert append_report_evaluations(combined,payload)==combined


def test_final_native_html_places_evidence_in_gapmer_tab(tmp_path):
    from aso_agents.html_report import render_goal_intake_html
    result=evaluate_candidates([row()],{},tmp_path,references=[reference()],predictor=predictor)
    html=render_goal_intake_html({"oligoai_evaluation":result})
    assert html.index('id="gapmer-content"') < html.index('id="oligoai-evaluation"') < html.index('id="steric-binding"')
    assert "__OLIGOAI_SECTION__" not in html


def test_physical_record_corruption_is_caught(tmp_path):
    r=row();r["synthesized_antisense_sequence"]="T"*20
    result=evaluate_candidates([r],{},tmp_path,references=[reference()],predictor=lambda *a:pytest.fail("corrupt molecule"))
    assert "contradicts" in result["candidates"][0]["reason"]


def test_deduplication_does_not_drop_candidate_or_reference_rows(tmp_path):
    seen=[]
    def observe(inputs,directory):seen.extend(inputs);return predictor(inputs,directory)
    result=evaluate_candidates([row(),row()],{},tmp_path,references=[reference(),{**reference(),"label":"other","transcript_id":"other"}],predictor=observe)
    assert len(seen)==1
    assert len(result["candidates"])==2
    assert all(len(c["contexts"])==2 for c in result["candidates"])


def test_supplied_table_metadata_uses_evaluated_molecule(tmp_path):
    native=row()
    supplied={"source":"provided", "sequence":SEQ, "platform":"gapmer", "input_row_id":"row_1",
              "orientation_interpretation":{"evaluated_molecule_sequence":SEQ},
              "input_metadata":{"molecule_specification":native["molecule_specification"], "IDT":native["chemistry_idt"]}}
    result=evaluate_candidates([supplied],{},tmp_path,references=[reference()],predictor=predictor)
    assert result["predicted_count"]==1
    assert result["candidates"][0]["candidate_id"]=="row_1"


def test_supplied_vendor_notation_without_a_record_is_attested(tmp_path):
    native=row()
    supplied={"source":"provided", "sequence":SEQ, "platform":"gapmer", "declared_sequence_role":"antisense_oligo_sequence",
              "input_metadata":{"IDT":native["chemistry_idt"]}}
    result=evaluate_candidates([supplied],{},tmp_path,references=[reference()],predictor=predictor)
    assert result["predicted_count"]==1


def test_missing_supplied_chemistry_does_not_borrow_goal_profile(tmp_path):
    supplied={"source":"provided","sequence":SEQ,"platform":"gapmer","declared_sequence_role":"antisense_oligo_sequence"}
    result=evaluate_candidates([supplied],{"chemistry":{"CHEMISTRY":"5-10-5","platform":"gapmer"}},tmp_path,
                               references=[reference()],predictor=lambda *a:pytest.fail("invented chemistry"))
    assert result["predicted_count"]==0


def test_explicit_prediction_conditions_cross_canonical_intake_boundaries():
    import jsonschema
    from aso_agents.goal_contract import canonical_structured_goal_schema,canonical_task_context_schema
    from aso_agents.pipeline_tools import _overlay_request_context_on_goal
    options={"dosage_nm":42,"transfection_method":"Gymnosis"}
    context={"oligoai_evaluation":options}
    jsonschema.validate(context,canonical_task_context_schema())
    goal=_overlay_request_context_on_goal({},context,"")
    assert goal["oligoai_evaluation"]==options
    field=canonical_structured_goal_schema()["properties"]["oligoai_evaluation"]
    jsonschema.validate(options,field)
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate({"transfection_method":"unknown"},field)


def test_final_supplied_report_contains_computed_prediction(tmp_path,monkeypatch):
    from aso_agents import pipeline_tools as p
    result=evaluate_candidates([row()],{},tmp_path,references=[reference()],predictor=predictor)
    monkeypatch.setattr(p,"_write_v2_excel_outputs",lambda *a:{})
    monkeypatch.setattr(p,"_write_requested_evaluation_exports",lambda *a:{})
    monkeypatch.setattr(p,"_chooser_focused_v2_markdown",lambda **k:"")
    artifacts=p._write_v2_response_outputs(run_dir=str(tmp_path),markdown_text="# ASO report",title="ASO report",
                                         response_payload={"evaluation_summary":{"oligoai_evaluation":result}},response_basename="existing_aso_evaluation")
    md=Path(artifacts["report_md"]).read_text();ht=Path(artifacts["report_html"]).read_text()
    assert "73.12" in md and "73.12" in ht
    assert md.count("## OligoAI gapmer efficacy prediction")==1


def model_input():
    r=row();record,sugars,bonds=model_chemistry(r,{})
    value={"antisense_5to3":SEQ,"physical_molecule_sha256":record["physical_molecule_sha256"],
           "sugar_mods":sugars,"backbone_mods":bonds,"rna_context":reference()["sequence"].replace("T","U"),
           "dosage_nm":100.,"transfection_method":"Lipofection"}
    value["input_id"]=digest(value)
    return value






def test_conditions_are_bound_to_the_producing_program(tmp_path):
    goal={"oligoai_evaluation":{"dosage_nm":100,"transfection_method":"Gymnosis"},
          "design_programs":[{"program_id":"one","oligoai_evaluation":{"dosage_nm":10}},
                             {"program_id":"two","oligoai_evaluation":{"dosage_nm":50}}]}
    first,second=row(),row();first["program_id"]="one";second["program_id"]="two"
    seen=[]
    def observe(inputs,directory):seen.extend(inputs);return predictor(inputs,directory)
    result=evaluate_candidates([first,second],goal,tmp_path,references=[reference()],predictor=observe)
    assert [i["dosage_nm"] for i in seen]==[10,50]
    assert all(i["transfection_method"]=="Gymnosis" for i in seen)
    assert result["predicted_count"]==2
    records=list(csv.DictReader(Path(result["artifact_csv"]).open()))
    assert [float(i["dosage_nm"]) for i in records]==[10,50]
    from run_goal_intake import _design_program_goal_copy
    child=_design_program_goal_copy(goal,goal["design_programs"][0])
    assert child["oligoai_evaluation"]=={"dosage_nm":10,"transfection_method":"Gymnosis"}


def test_report_evidence_links_are_portable(tmp_path):
    result=evaluate_candidates([row()],{},tmp_path,references=[reference()],predictor=predictor)
    markdown=markdown_section(result,report_dir=tmp_path)
    html=html_section(result,report_dir=tmp_path)
    assert str(tmp_path) not in markdown and str(tmp_path) not in html
    assert 'href="oligoai/' in html
    assert 'evaluation.json"' in html


def test_missing_genomic_contig_keeps_attested_mature_fallback():
    runtime=SimpleNamespace(_resolve_patient_reference_fasta=lambda goal:None)
    catalog={"target":{"chrom":"chrMissing","strand":"+","exons_tx":[{"start":1,"end":20}]}}
    refs=target_references({},runtime,catalog=catalog,cdna={"target":reverse_complement(SEQ)},genomes={"target":{}})
    assert len(refs)==1 and refs[0]["kind"]=="mature_rna"


def test_unsupported_profile_still_displays_the_physical_sequence(tmp_path):
    result=evaluate_candidates([row(architecture="2ome:5-10-5")],{},tmp_path,references=[reference()])
    assert SEQ in markdown_section(result)
    assert SEQ in html_section(result)


@pytest.mark.parametrize('options', [{}, {'dosage_nm': 42, 'transfection_method': 'Gymnosis'}])
def test_repeated_report_publication_preserves_prediction_inputs(tmp_path, options):
    from aso_agents.pipeline_tools import _publish_generated_artifacts
    goal = {'oligoai_evaluation': deepcopy(options)}
    final = {}
    for generation in range(3):
        candidates = [row()]
        result = evaluate_candidates(candidates, goal, tmp_path, references=[reference()], predictor=predictor)
        assert result['status'] == 'complete'
        assert result['predicted_count'] == 1
        _publish_generated_artifacts(goal, final, {'oligoai_evaluation': result,
                                                 'generated_aso_set_json': str(tmp_path/'candidates.json')})
        assert goal['oligoai_evaluation'] == options
        assert final['oligoai_evaluation']['predicted_count'] == 1
        assert evaluation_conditions(goal) == result['conditions']
        assert '73.12' in markdown_section(final['oligoai_evaluation'])


def test_computed_artifacts_cannot_replace_any_canonical_request_field():
    from aso_agents.pipeline_tools import _publish_generated_artifacts
    goal = {'chemistry': {'CHEMISTRY': '5-10-5'}, 'candidate_limit': 12}
    original = deepcopy(goal)
    final = {}
    artifacts = {'chemistry': {'output_summary': 'computed'}, 'candidate_limit': 4,
                 'generated_aso_set_json': 'candidates.json'}
    _publish_generated_artifacts(goal, final, artifacts)
    assert goal['chemistry'] == original['chemistry']
    assert goal['candidate_limit'] == original['candidate_limit']
    assert goal['generated_aso_set_json'] == 'candidates.json'
    assert final == artifacts
