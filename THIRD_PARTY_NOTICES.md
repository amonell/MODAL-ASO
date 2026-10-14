# Third-party acknowledgments

MODAL-ASO includes a modified ASO tiling workflow derived from **ASO-designer**, developed by Brian Yee and the Yeo laboratory. Its maintainers have authorized inclusion in this GPL-3.0 distribution. The workflow is in `aso_agents/resources/tiling/`.

The following projects provide separately installed tools or models:

| Project | Contribution | Upstream license |
| --- | --- | --- |
| [SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer) | Human splice prediction | Apache-2.0 |
| [OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI) | Species-specific splice prediction | GPL-3.0 |
| [OligoAI](https://github.com/barneyhill/OligoAI) | Gapmer efficacy prediction | Apache-2.0 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | OligoAI attention kernels | BSD-3-Clause |
| [Bowtie](https://github.com/BenLangmead/bowtie) | Sequence alignment | Artistic-2.0 |
| [BEDTools](https://github.com/arq5x/bedtools2) | Genomic interval operations | MIT |
| [SAMtools](https://github.com/samtools/samtools) | Sequence and alignment files | MIT / Expat |
| [Subread](https://subread.sourceforge.net/) | Feature assignment | GPL-3.0 |
| [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) | RNA structure and accessibility | ViennaRNA license |
| [UCSC utilities](https://genome.ucsc.edu/license/) | Browser-track exports | See upstream terms |

Model checkpoints and reference data are downloaded from their upstream sources. Their original terms continue to apply. Source revisions, checkpoint hashes and environment pins are included with the installer; upstream license files remain in installed model checkouts.

Reference data are provided by NCBI, GENCODE, Ensembl, UCSC, ENCODE, GTEx, CELLxGENE and TargetScan as required by the workflow. Please also cite the data and models used in your analysis.
