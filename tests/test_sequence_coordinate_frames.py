import json
from itertools import product

import numpy as np
import pytest

from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.sequence_coordinates import SequenceCoordinates


@pytest.mark.parametrize("length", [1, 2, 7, 19, 23, 37])
@pytest.mark.parametrize("first,direction", product([-41, -1, 0, 1, 1053], [-1, 1]))
def test_all_bounded_intervals_against_explicit_coordinate_labels(length, first, direction):
    # The oracle labels each character explicitly and selects by interval
    # membership, rather than repeating the implementation's slice arithmetic.
    sequence = "".join(chr(0x400 + i) for i in range(length))
    labels = list(range(first, first + direction * length, direction))
    frame = SequenceCoordinates(first_coordinate=first, length=length, direction=direction)
    for index, coordinate in enumerate(labels):
        assert frame.to_index(coordinate) == index
        assert frame.to_coordinate(index) == coordinate
    for a, b in product(sorted(labels), repeat=2):
        if a > b:
            continue
        expected = "".join(base for label, base in zip(labels, sequence) if a <= label <= b)
        assert frame.extract(sequence, a, b) == expected
        assert sequence[frame.inclusive_slice(a, b)] == expected
        assert len(expected) == b - a + 1


@pytest.mark.parametrize("values", [
    {"first_coordinate": True, "length": 5},
    {"first_coordinate": 0.0, "length": 5},
    {"first_coordinate": 0, "length": False},
    {"first_coordinate": 0, "length": 0},
    {"first_coordinate": 0, "length": -1},
    {"first_coordinate": 0, "length": 5.0},
    {"first_coordinate": 0, "length": 5, "direction": 0},
    {"first_coordinate": 0, "length": 5, "direction": True},
    {"first_coordinate": 0, "length": 5, "direction": "-1"},
])
def test_invalid_frames_are_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        SequenceCoordinates(**values)


@pytest.mark.parametrize("direction", [-1, 1])
def test_out_of_bounds_and_wrong_sequence_never_silently_clip(direction):
    frame = SequenceCoordinates(first_coordinate=-7, length=5, direction=direction)
    coordinates = [-7 + direction * i for i in range(5)]
    low, high = min(coordinates), max(coordinates)
    for index in [-1, 5]:
        with pytest.raises(ValueError):
            frame.to_coordinate(index)
    for coordinate in [low - 1, high + 1]:
        with pytest.raises(ValueError):
            frame.to_index(coordinate)
    for bounds in [(low - 1, high), (low, high + 1), (high, low)]:
        with pytest.raises(ValueError):
            frame.inclusive_slice(*bounds)
    for sequence in ["ACGT", "ACGTAC"]:
        with pytest.raises(ValueError, match="length"):
            frame.extract(sequence, low, high)
    for value in [True, 1.0, "1", None]:
        with pytest.raises(TypeError):
            frame.to_index(value)
        with pytest.raises(TypeError):
            frame.to_coordinate(value)


def test_numpy_integers_are_normalized_and_frame_is_immutable():
    from dataclasses import FrozenInstanceError, asdict
    frame = SequenceCoordinates(first_coordinate=np.int64(0), length=np.int32(3), direction=np.int64(-1))
    assert json.loads(json.dumps(asdict(frame))) == {"first_coordinate": 0, "length": 3, "direction": -1}
    assert frame.extract("ACG", -2, 0) == "ACG"
    with pytest.raises(FrozenInstanceError):
        frame.first_coordinate = 99


def test_actual_analysis_runtime_preserves_both_inclusive_endpoints(tmp_path):
    destination = tmp_path / "comparisons.json"
    (tmp_path / "declared_inputs.json").write_text(json.dumps({
        "sequence": "ACGTAGC", "other_sequence": "ACGTACT", "first_coordinate": -4,
    }))

    def generate(payload):
        contract = payload["execution_contract"]
        return {"python_code": "\n".join([
            "import json", "from pathlib import Path",
            "from aso_agents.sequence_coordinates import SequenceCoordinates",
            "from aso_agents.mapping_evidence import compare_antisense_target",
            "inputs=json.loads(Path('declared_inputs.json').read_text())",
            "frame=SequenceCoordinates(first_coordinate=inputs['first_coordinate'], length=len(inputs['sequence']))",
            "target=frame.extract(inputs['other_sequence'], -2, 2)",
            "comparison=compare_antisense_target('GCTAC', target)",
            "result={'target':target,'comparison':comparison}",
            f"Path({str(destination)!r}).write_text(json.dumps(result))",
            f"Path({contract['result_json']!r}).write_text(json.dumps(result))",
            f"Path({contract['summary_markdown']!r}).write_text('Both target endpoints included.')",
        ])}

    outcome = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                       required_outputs=[destination], max_attempts=1)
    assert outcome["ok"], outcome
    result = json.loads(destination.read_text())
    assert result["target"] == "GTACT"
    assert result["comparison"]["mismatch_count"] == 2
    assert result["comparison"]["mismatch_positions_in_supplied_sequence_1based"] == [1, 2]
