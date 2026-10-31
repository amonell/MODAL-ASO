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
