import numpy as np
import pytest

from aso_agents.cryptic_splice_evidence import annotation_sites, unannotated_gain


@pytest.mark.parametrize("strand",["+","-"])
def test_gain_coordinates_annotations_and_perturbed_span_on_both_strands(strand):
    ref = np.zeros((11,3))
    alt = ref.copy()
    def index(position): return position-100 if strand=="+" else 110-position
    alt[index(102),1] = 0.9  # annotated, excluded
    alt[index(105),1] = 0.8  # ASO span, excluded
    alt[index(108),1] = 0.4
    ref[index(108),1] = 0.1
    alt[index(109),2] = 0.25
    value = unannotated_gain(ref,alt,chrom="chr1",left=100,right=110,strand=strand,
        annotated={"acceptor":[102],"donor":[]},annotation_radius_nt=0,excluded_intervals=[(104,106)])
    assert value["status"] == "scanned_within_declared_scope"
    assert value["sites"]["acceptor"]["largest_gain"]["position_1based"] == 108
    assert value["sites"]["acceptor"]["largest_gain"]["delta"] == pytest.approx(0.3)
    assert value["sites"]["donor"]["largest_gain"]["position_1based"] == 109
    assert value["sites"]["acceptor"]["scanned_position_count"] == 7


def test_missing_nonfinite_and_empty_scope_never_become_a_clean_screen():
    kwargs = dict(chrom="chr1",left=100,right=110,strand="+",annotated={})
    ref = np.zeros((11,3))
    assert unannotated_gain(ref,ref[:10],**kwargs)["status"] == "unavailable"
    alt = ref.copy(); alt[2,1] = np.nan
    assert unannotated_gain(ref,alt,**kwargs)["status"] == "incomplete"
    assert unannotated_gain(ref,ref,excluded_intervals=[(100,110)],**kwargs)["status"] == "no_scannable_positions"
    result = unannotated_gain(ref,ref,**kwargs)
    assert result["sites"]["acceptor"]["largest_gain"] is None
    assert result["sites"]["acceptor"]["scanned_position_count"] == 11


def test_annotation_scope_includes_noncoding_boundaries_not_transcript_ends():
    exons = [{"start":100,"end":150},{"start":200,"end":250},{"start":300,"end":350}]
    assert annotation_sites(exons,"+") == {"acceptor":[200,300],"donor":[150,250]}
    assert annotation_sites(exons,"-") == {"acceptor":[150,250],"donor":[200,300]}


@pytest.mark.parametrize("script", ["scripts/run_openspliceai_eval.py", "aso_agents/resources/scripts/run_openspliceai_eval.py"])
def test_worker_preserves_metrics_and_emits_scoped_evidence_without_harness_import(script, monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path
    import sys
    import types
    native_import = builtins.__import__
    def isolated_import(name, *args, **kwargs):
        if name == "aso_agents" or name.startswith("aso_agents."):
            raise AssertionError("Separate model interpreter must not import harness dependencies")
        return native_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", isolated_import)
    predictor = types.ModuleType("openspliceai.predict")
    predictor.predict = types.SimpleNamespace()
    predictor.utils = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openspliceai", types.ModuleType("openspliceai"))
    monkeypatch.setitem(sys.modules, "openspliceai.predict", predictor)
    path = Path(__file__).parents[1] / script
    spec = importlib.util.spec_from_file_location("native_splice_worker_control", path)
    worker = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, worker)
    spec.loader.exec_module(worker)
    class Model:
        def __init__(self, *args): pass
        def predict(self, sequence):
            scores = np.zeros((len(sequence), 3))
            if sequence.startswith("N"):
                scores[8, 1] = 0.4
            elif sequence.startswith("T"):
                scores[9, 2] = 0.2
            return scores
    monkeypatch.setattr(worker, "OpenSpliceAIModel", Model)
    request = {"id":"control", "reference_sequence":"A"*11,
               "deletion_sequence":"N"+"A"*10, "replacement_sequence":"T"+"A"*10,
               "primary":{"acceptor_index":2,"donor_index":7,"window":1}}
    payload = {"model_path":"unused", "requests":[request]}
    base = worker.evaluate_requests(*worker._parse_requests(payload))["results"][0]
    request["cryptic_context"] = dict(chrom="chr1",left=100,right=110,strand="+",
        annotated={"acceptor":[102],"donor":[107]},excluded_intervals=[(100,101)])
    result = worker.evaluate_requests(*worker._parse_requests(payload))["results"][0]
    assert result["metrics"] == base["metrics"]
    evidence = result["cryptic_splice_evidence"]
    assert evidence["first_perturbation"] == "length_preserving_N_mask"
    assert evidence["deletion"]["sites"]["acceptor"]["largest_gain"]["position_1based"] == 108
    assert evidence["replacement"]["sites"]["donor"]["largest_gain"]["position_1based"] == 109
