"""An empty mapping search is a valid empty result, not a pipeline exception."""

import subprocess
import sys
from pathlib import Path


def test_filter_mapped_writes_empty_fasta_when_no_queries_map(tmp_path):
    script = (Path(__file__).resolve().parents[1] /
              "aso-designer-main/aso-designer-main/filter_mapped.py")
    sam = tmp_path / "mapped.sam"
    sam.write_text("@HD\tVN:1.6\tSO:unsorted\n")
    features = tmp_path / "featurecounts.txt"
    features.write_text("no_hit\t0\t0\tNone\n")
    output = tmp_path / "mapped.fasta"
    result = subprocess.run([sys.executable, str(script), "-s", str(sam),
                             "-f", str(features), "-o", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert output.read_text() == ""
