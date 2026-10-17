"""Coordinate-domain errors must be actionable without silently remapping input."""
import json

import pytest

from aso_agents.analysis_helpers import shared_analysis_helpers
from aso_agents.spliced_coordinates import SplicedSequenceCoordinates


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("operation", ["project", "extract", "to_genomic"])
def test_mature_bounds_error_exposes_input_domain_and_explicit_inverse(strand, operation):
    frame = SplicedSequenceCoordinates(chromosome="contig", strand=strand,
                                       exons=[(101, 105), (201, 205)])
    def forbidden(*args):
        raise AssertionError("An invalid mature interval must not fetch sequence")
    with pytest.raises(ValueError) as caught:
        if operation == "project":
            frame.project_interval(201, 201)
        elif operation == "extract":
            frame.extract(201, 201, fetch_genomic=forbidden)
        else:
            frame.to_genomic(201)
    message = str(caught.value)
    assert "start1=201, end1=201" in message
    assert "mature/cDNA" in message and "1 <= start1 <= end1 <= 10" in message
    assert "to_mature(genomic_position1)" in message
    assert "No coordinates were converted or clipped" in message
    # The explicitly requested inverse produces a valid projection on either strand.
    mature = frame.to_mature(201)
    assert mature == (6 if strand == "+" else 5)
    assert frame.project_interval(mature, mature)[0]["genomic_start1"] == 201


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("position", [6, 150, 300])
def test_invalid_genomic_position_stays_rejected_with_both_domains_identified(strand, position):
    frame = SplicedSequenceCoordinates(chromosome="contig", strand=strand,
                                       exons=[(101, 105), (201, 205)])
    with pytest.raises(ValueError) as caught:
        frame.to_mature(position)
    message = str(caught.value)
    assert f"genomic_position1={position} on contig ({strand} strand)" in message
    assert "one-based genomic exon position" in message
    assert "mature/cDNA position in 1..10" in message
    assert "to_genomic(mature_position1)" in message


def test_advertised_helper_distinguishes_query_domain_from_fetch_callback_domain():
    helper = next(h for h in shared_analysis_helpers() if h["name"] == "SplicedSequenceCoordinates")
    contract = helper["returns"]
    assert "project_interval(start1,end1) takes mature/cDNA endpoints in 1..length" in contract
    assert "extract(start1,end1,fetch_genomic=callable) also takes mature/cDNA endpoints" in contract
    assert "callback receives plus-genomic bounds" in contract
    assert "to_mature(genomic_position1)" in contract and "to_genomic(mature_position1)" in contract


def test_existing_execution_repair_receives_actual_coordinate_diagnosis(tmp_path):
    from aso_agents.analysis_execution import execute_analysis_program
    prefix = ("from aso_agents.spliced_coordinates import SplicedSequenceCoordinates\n"
              "frame = SplicedSequenceCoordinates(chromosome='contig', strand='-', exons=[(101,105),(201,205)])\n")
    original = prefix + "blocks = frame.project_interval(201, 201)\n"
    seen = []
    def generate(payload):
        seen.append(payload)
        if len(seen) == 1:
            return {"python_code": original}
        diagnostic = payload["execution_repair"]["runtime_diagnostics"]
        assert diagnostic["exception_type"] == "ValueError"
        assert "start1=201, end1=201" in diagnostic["message"]
        assert "1 <= start1 <= end1 <= 10" in diagnostic["message"]
        assert "to_mature(genomic_position1)" in diagnostic["message"]
        return {"python_code": prefix +
                "mature = frame.to_mature(201)\nblocks = frame.project_interval(mature, mature)\n"
                "from pathlib import Path\nimport json\n"
                "Path('sandbox_result.json').write_text(json.dumps({'blocks':blocks}))\n"
                "Path('sandbox_summary.md').write_text('Projected the supplied exonic genomic base through its mature coordinate.')\n"}
    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[], max_attempts=2)
    assert result["ok"] and len(result["attempts"]) == len(seen) == 2
    assert result["result"]["blocks"][0]["mature_start1"] == 5
    assert result["result"]["blocks"][0]["genomic_start1"] == 201
    assert (tmp_path / "execution_attempts/01/generated_code.py").read_text() == original.strip()
    diagnostic = json.loads((tmp_path / "execution_attempts/01/runtime_error.json").read_text())
    assert diagnostic["exception_type"] == "ValueError"
