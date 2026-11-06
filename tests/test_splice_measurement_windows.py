from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from aso_agents.splice_measurement_windows import delta_metrics, separate_measurement_windows


def site(position, strand="+", kind="acceptor", chrom="chr8"):
    return {"site_kind": kind, "window_bp": 10,
            kind: {"chrom": chrom, "position": position, "strand": strand,
                   "start": position - 10, "end": position + 10, "window_bp": 10}}


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("kind", ["acceptor", "donor"])
@pytest.mark.parametrize("distance,radius", [(1, 0), (2, 0), (3, 1), (20, 9), (21, 10)])
def test_close_junctions_have_disjoint_measurement_windows(strand, kind, distance, radius):
    sites = [site(100, strand, kind), site(100 + distance, strand, kind), site(100, strand, kind)]
    before = deepcopy(sites)
    result = separate_measurement_windows(sites)
    assert sites == before
    assert [s[kind]["window_bp"] for s in result] == [radius] * 3
    assert result[0][kind]["end"] < result[1][kind]["start"]
    assert result[0] == result[2]
    assert all(s["window_bp"] == radius for s in result)


def test_separate_channels_chromosomes_and_strands_keep_requested_radius():
    sites = [site(100), site(101, kind="donor"), site(101, chrom="chr9"), site(101, strand="-")]
    assert separate_measurement_windows(sites) == sites


def test_acceptor_changes_preserve_shared_donor_measurements():
    sites = [site(100), site(103)]
    for s in sites:
        s["donor"] = site(150, kind="donor")["donor"]
    result = separate_measurement_windows(sites)
    assert [s["acceptor"]["window_bp"] for s in result] == [1, 1]
    assert [s["donor"]["window_bp"] for s in result] == [10, 10]
    reference = np.zeros((80, 3)); alternative = reference.copy()
    reference[20, 1], reference[23, 1], reference[70, 2] = .8, .2, .7
    alternative[20, 1], alternative[23, 1], alternative[70, 2] = .1, .9, .6
    measured = [delta_metrics(reference, alternative, pos, 70, 1, 10) for pos in (20, 23)]
    assert [m["acceptor"] for m in measured] == pytest.approx([-.7, .7])
    assert [m["donor"] for m in measured] == pytest.approx([-.1, -.1])
    assert measured[0]["pair"] != measured[1]["pair"]


def test_exact_and_unavailable_measurements_remain_distinct():
    reference = np.zeros((10, 3)); alternative = reference.copy(); alternative[3, 1] = .8
    assert delta_metrics(reference, alternative, 2, 6, 0)["acceptor"] == 0
    assert delta_metrics(reference, alternative, 3, 6, 0)["acceptor"] == .8
    assert delta_metrics(reference, alternative, -1, 6, 1) == {"pair": None, "acceptor": None, "donor": 0.0}


@pytest.mark.parametrize("script", ["scripts/run_openspliceai_eval.py", "aso_agents/resources/scripts/run_openspliceai_eval.py"])
def test_worker_preserves_independent_windows_including_exact_position(script, monkeypatch):
    predictor = types.ModuleType("openspliceai.predict")
    predictor.predict = types.SimpleNamespace(); predictor.utils = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openspliceai", types.ModuleType("openspliceai"))
    monkeypatch.setitem(sys.modules, "openspliceai.predict", predictor)
    spec = importlib.util.spec_from_file_location("window_test_worker", Path(__file__).parents[1] / script)
    worker = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, worker); spec.loader.exec_module(worker)
    class Model:
        def __init__(self, *args): pass
        def predict(self, sequence):
            scores = np.zeros((20, 3))
            if sequence[0] == "A": scores[4, 1], scores[5, 1] = .8, .2
            else: scores[4, 1], scores[5, 1], scores[15, 2] = .1, .9, .4
            return scores
    monkeypatch.setattr(worker, "OpenSpliceAIModel", Model)
    windows = [{"key": str(i), "acceptor_index": i, "donor_index": 15,
                "window": 10, "acceptor_window": 0, "donor_window": 2} for i in (4, 5)]
    payload = {"model_path": "unused", "requests": [{"id": "test", "reference_sequence": "A" * 20,
        "deletion_sequence": "N" * 20, "replacement_sequence": "T" * 20,
        "sites": windows, "primary": windows[1]}]}
    result = worker.evaluate_requests(*worker._parse_requests(payload))["results"][0]
    assert [s["deletion"]["acceptor"] for s in result["sites"]] == pytest.approx([-.7, .7])
    assert [s["deletion"]["donor"] for s in result["sites"]] == pytest.approx([.4, .4])
    assert result["metrics"]["deletion"] == result["sites"][1]["deletion"]
