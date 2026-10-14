"""Check runtime resources and the public scope of a built wheel."""
from pathlib import Path
import sys
import zipfile

wheels = list(Path(sys.argv[1]).glob("*.whl"))
assert len(wheels) == 1, wheels
with zipfile.ZipFile(wheels[0]) as archive:
    names = archive.namelist()
    required = ["aso_agents/cli.py", "aso_agents/encode_rbp_peaks.py", "aso_chemistry.py",
                "aso_agents/resources/tiling/Snakefile", "aso_agents/resources/tiling/filter_mapped.py",
                "aso_agents/resources/tiling/filter_seqs.py", "aso_agents/resources/tiling/tile_seqs.py",
                "aso_agents/resources/scripts/download_ensembl_reference_assets.py",
                "aso_agents/resources/scripts/run_openspliceai_eval.py",
                "aso_agents/resources/requirements/oligoai.txt",
                "aso_agents/resources/requirements/tools-linux-64.txt",
                "aso_agents/resources/requirements/tools-osx-arm64.txt"]
    assert all(name in names for name in required), set(required)-set(names)
    forbidden = ("/benchmark/", "/evals/", "__pycache__", "AGENTS.md", ".pyc", "figures/")
    assert not any(any(value in name for value in forbidden) for name in names)
print("Wheel resources and scope verified:", wheels[0].name)
