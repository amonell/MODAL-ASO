import hashlib
import json
from itertools import product

import pytest

from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.sequence_coordinates import SequenceCoordinates


@pytest.mark.parametrize("bounds", [(-8, -3), (-3, 3), (0, 0), (11, 18)])
@pytest.mark.parametrize("direction", [-1, 1])
def test_declared_bounds_preserve_every_labeled_base_and_reject_changed_length(bounds, direction):
    lower, upper = bounds
    labels = list(range(lower, upper + 1))[::direction]
    sequence = "".join(chr(0x400 + i) for i in range(len(labels)))
    frame = SequenceCoordinates.from_inclusive_bounds(lower=lower, upper=upper, direction=direction)
    for a, b in product(range(lower, upper + 1), repeat=2):
        if a <= b:
            expected = "".join(base for base, label in zip(sequence, labels) if a <= label <= b)
            assert frame.extract(sequence, a, b) == expected
    for incorrect in [sequence[1:], sequence + "A"]:
        with pytest.raises(ValueError, match="differs from frame length"):
            frame.extract(incorrect, lower, upper)


@pytest.mark.parametrize("values", [
    {"lower": True, "upper": 5, "direction": 1},
    {"lower": 1, "upper": 5.0, "direction": 1},
    {"lower": 1, "upper": 5, "direction": False},
    {"lower": 1, "upper": 5, "direction": 0},
    {"lower": 1, "upper": 5, "direction": 2},
    {"lower": 5, "upper": 1, "direction": -1},
])
def test_malformed_bounds_are_not_inferred_or_reordered(values):
    with pytest.raises((TypeError, ValueError)):
        SequenceCoordinates.from_inclusive_bounds(**values)


def test_execution_repair_rereads_original_sequence_without_changing_declared_bounds(tmp_path):
    source = tmp_path / "declared_inputs.json"
    source.write_text(json.dumps({"sequence": "GATCGAC", "lower": -3, "upper": 3, "direction": -1}))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "window.json"
    prefix = ("import json\nfrom pathlib import Path\n"
              "from aso_agents.sequence_coordinates import SequenceCoordinates\n"
              "source = json.loads(Path('declared_inputs.json').read_text())\n"
              "frame = SequenceCoordinates.from_inclusive_bounds(lower=source['lower'], "
              "upper=source['upper'], direction=source['direction'])\n")
    suffix = ("frame.extract(sequence, source['lower'], source['upper'])\n"
              "result = {'window':frame.extract(sequence, -1, 2)}\n"
              f"Path({str(destination)!r}).write_text(json.dumps(result))\n"
              "Path('sandbox_result.json').write_text(json.dumps(result))\n"
              "Path('sandbox_summary.md').write_text('Extracted from the unchanged source bounds.')\n")
    first = prefix + "sequence = source['sequence'][1:]\n" + suffix
    calls = []

    def generate(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"python_code": first}
        assert len(calls) == 2 and not destination.exists()
        diagnostic = payload["execution_repair"]["runtime_diagnostics"]
        assert diagnostic["exception_type"] == "ValueError"
        assert "Sequence length 6 differs from frame length 7" in diagnostic["message"]
        return {"python_code": prefix + "sequence = source['sequence']\n" + suffix}

    outcome = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                       required_outputs=[destination], max_attempts=2)
    assert outcome["ok"] and len(outcome["attempts"]) == len(calls) == 2
    assert json.loads(destination.read_text()) == {"window": "ATCG"}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert (tmp_path / "execution_attempts/01/generated_code.py").read_text() == first.strip()
    assert json.loads((tmp_path / "execution_attempts/01/runtime_error.json").read_text())["exception_type"] == "ValueError"
