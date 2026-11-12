"""Check runtime resources and the public scope of a built wheel."""
from pathlib import Path
from email.parser import BytesParser
import sys
import tarfile
import tomllib
import zipfile

version = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]["version"]
wheels = list(Path(sys.argv[1]).glob("*.whl"))
assert len(wheels) == 1, wheels
assert wheels[0].name == f"modal_aso-{version}-py3-none-any.whl", wheels[0].name
with zipfile.ZipFile(wheels[0]) as archive:
    names = archive.namelist()
    metadata = BytesParser().parsebytes(archive.read(f"modal_aso-{version}.dist-info/METADATA"))
    assert metadata["Name"] == "modal-aso" and metadata["Version"] == version
    required = ["aso_agents/cli.py", "aso_agents/encode_rbp_peaks.py", "aso_chemistry.py",
                "aso_agents/safeguard_review.py",
                "aso_agents/splice_recovery.py",
                "aso_agents/splice_measurement_windows.py",
                "aso_agents/report_tables.py", "aso_agents/transcript_rules.py",
                "aso_agents/resources/tiling/Snakefile", "aso_agents/resources/tiling/filter_mapped.py",
                "aso_agents/resources/tiling/recover_overflow.py",
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

sdists = list(Path(sys.argv[1]).glob("*.tar.gz"))
assert len(sdists) <= 1, sdists
if sdists:
    assert sdists[0].name == f"modal_aso-{version}.tar.gz", sdists[0].name
    with tarfile.open(sdists[0]) as archive:
        prefix = f"modal_aso-{version}/"
        required_source = ["uv.lock", "CITATION.cff", "CHANGELOG.md", "CONTRIBUTING.md", "ci/check_version.py"]
        names = set(archive.getnames())
        assert all(prefix + name in names for name in required_source), required_source
        metadata = BytesParser().parsebytes(archive.extractfile(f"modal_aso-{version}/PKG-INFO").read())
        assert metadata["Name"] == "modal-aso" and metadata["Version"] == version
    print("Source distribution version verified:", sdists[0].name)
