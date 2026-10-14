"""Exercise wheel resources without adding the source checkout to sys.path."""

import argparse
import json
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace

from aso_agents import tiling_agent
from aso_agents.paths import resource_root
from aso_agents.runtime_binaries import refresh_runtime_path
import aso_chemistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    for location in (Path(tiling_agent.__file__), Path(aso_chemistry.__file__), resource_root()):
        assert not location.resolve().is_relative_to(args.source_root.resolve()), location
    refresh_runtime_path()
    with tempfile.TemporaryDirectory(prefix="modal-aso installed ") as work:
        root = Path(work)
        reference = root / "reference with spaces"
        reference.mkdir()
        rng = random.Random(401)
        sequence = "".join(rng.choices("ACGT", k=3000))
        fasta = reference / "neutral.fa"
        fasta.write_text(">chr1\n" + sequence + "\n")
        gtf = reference / "neutral.gtf"
        gtf.write_text('chr1\tfixture\texon\t1\t3000\t.\t+\t.\tgene_id "gene1"; transcript_id "tx1"; gene_name "NEUTRAL";\n')
        sizes = reference / "neutral.sizes"
        sizes.write_text("chr1\t3000\n")
        prefix = reference / "neutral"
        subprocess.run(["bowtie-build", str(fasta), str(prefix)], check=True, capture_output=True)
        cfg = SimpleNamespace(genome_fasta=fasta, phastcons_path=reference/"unused.bw",
                              phylop_path=reference/"unused2.bw", chrom_sizes_path=sizes,
                              bowtie_prefix=prefix, gtf_path=gtf, conservation_mode="disabled", ucsc_build="hg38")
        original = tiling_agent.get_species_config
        try:
            tiling_agent.get_species_config = lambda *_: cfg
            output = root / "results with spaces"
            manifest = tiling_agent.emit_pipeline_config(
                json.dumps({"gene_symbol": "NEUTRAL", "species": "human", "gc_min": 0, "gc_max": 1}),
                json.dumps({"rows": [["chr1", 400, 500, "neutral_region", 0, "+"]]}),
                json.dumps({"chemistry": "5-10-5"}), output_dir=str(output))
            result = subprocess.run(manifest["snakemake_cmd"], shell=True, text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
            result = tiling_agent.collect_results_json(str(output), species="human")
            rows = json.loads(Path(result["json_path"]).read_text())["rows"]
            assert len(rows) == 81, len(rows)
        finally:
            tiling_agent.get_species_config = original
    print("Installed wheel: native tiling passed outside the source checkout")


if __name__ == "__main__":
    main()
