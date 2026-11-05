# References and storage

## Choose the data directory

```bash
modal-aso setup --data-dir ~/modal-aso-data --species human --features tools references splice
```

This saves the location for future sessions. Use a persistent drive with room for genomes, conservation tracks, indexes and models.

Defaults:

| Location | Linux / WSL2 | macOS |
| --- | --- | --- |
| Data | `~/.local/share/modal-aso` | `~/Library/Application Support/MODAL-ASO` |
| Cache | `~/.cache/modal-aso` | `~/Library/Caches/MODAL-ASO` |
| Run outputs | `./runs` | `./runs` |

`MODAL_ASO_DATA_DIR` overrides the saved data location. `MODAL_ASO_CACHE_DIR` selects a cache directory. `ASO_REFERENCE_DATA_ROOT` selects a separate reference directory. Linux also honors `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `XDG_CONFIG_HOME`.

## Install references

```bash
modal-aso setup --species human mouse --features tools references splice
```

| Species | Assembly | Annotation | Conservation |
| --- | --- | --- | --- |
| Human | GRCh38 no-alt | GENCODE v42 | UCSC 30-way |
| Mouse | GRCm39 | GENCODE vM35 | UCSC 35-way |

Static references use pinned checksums; verified mirrors support interrupted transfers. Downloads receive integrity records. Extraction and Bowtie indexing happen once. Repeated setup verifies and reuses completed assets. Incomplete transfers remain separate from ready files.

Other species use Ensembl reference retrieval during the relevant workflow. Splice prediction requires a compatible species model; [model configuration](llm-providers.md) describes language-model settings separately.

## Missing bases during off-target screening

The human alignment reference masks some chromosome Y sequence. Gene-level off-target screening checks every annotated exon for missing bases, including exon interiors. When needed, MODAL-ASO automatically downloads the matching GENCODE transcript FASTA: v42 for human or vM35 for mouse. These checksum-pinned reference files use about 79 MiB or 54 MiB, respectively, in `MODAL_ASO_CACHE_DIR/reference_transcripts`. They stay outside report bundles. Users do not need previous run outputs or a supplied evidence cache.

Recovery requires matching transcript and gene versions, matching exon lengths, and agreement with every available genomic base. The recovered exon windows and adjacent exon junctions are searched at the requested mismatch budget. Remaining unknown bases are checked against each ASO: a gap affects completeness only when it could conceal a match within that budget. This also applies to exact-match searches.

A failed download or inconsistent reference leaves the affected search incomplete. Complete counts stay unavailable until the missing check is resolved; observed matches and the failure reason remain in the result. Other annotations retain this behavior when a matching transcript reference is unavailable.

A design can finish with incomplete off-target screening. Reports retain the screening status, missing counts and reference details for each requested species, and the completion record includes screening warnings. Child designs carry these details into composite reports. Missing counts remain unknown. Evaluation tasks that explicitly disable off-target screening retain that scope.

Genomic tiling automatically repeats searches that exceed the initial 2,000-match limit. Recovery requests every placement, counts distinct loci on disk and searches duplicate sequences once. Each original query retains its own intended-site exclusion. Counts and gene annotations cover the full search; location previews show up to 100 matches per mismatch group. Recovery receipts record the query digest, reference index, search command, alignment-stream digest and completion status. If recovery fails, the design still finishes with the affected screening counts marked unknown.

## Import existing references

```bash
modal-aso assets import --species human /path/to/existing/reference-directory
```

The directory should contain the matching FASTA, GTF, chromosome sizes, conservation tracks and RefSeq mapping files. Run `modal-aso assets list --species human` to see the expected filenames.

Import verifies the assembly and annotation, then links files into the managed data directory. Existing Bowtie indexes are checked against the genome before reuse. A missing index is built once.

The source directory can be read-only. It must remain accessible while the links are in use. Keep the managed data directory writable for indexes, locks and installation records.

## Supplied isoform files

Name the requested isoform and provide its SnapGene or GenBank path in the request. Quote paths containing spaces. Each requested isoform outcome retains its own file binding; a file listed as an attachment alone does not replace that biological objective. Structured requests use `artifact_ref_path` on the corresponding isoform objective and primary-outcome target.

Whether the supplied sequence also defines a patient genetic background is a separate choice. The harness asks for clarification when that role is unresolved. Missing files or unresolved annotations stop reference preparation instead of silently removing the requested outcome.

A quoted instruction about increasing or decreasing an isoform does not choose its genetic background. Free-text requests need explicit background or annotation-only intent; a model-inferred Boolean is insufficient. Ambiguous or negated quotes trigger the per-file question. A later clear answer is bound to the file's sequence identity, and a changed sequence requires a new choice.

Verified external accessions retain their mapping provenance when joined to the physical sequence catalog. The resolved transcript version and exon structure must agree; matching a stable accession alone does not erase a version conflict.

## Check stored files

```bash
modal-aso assets list --species human mouse
modal-aso assets verify --species human mouse --models
```

`verify` recomputes file hashes. Normal setup uses stored hashes and file metadata to avoid repeatedly reading large files.

## Choose where runs go

```bash
modal-aso --output-dir ~/aso-results "your request"
```

Each request receives its own run directory. API calls and online database queries may still need network access after asset setup.
