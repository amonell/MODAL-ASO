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

Downloads receive integrity records. Extraction and Bowtie indexing happen once. Repeated setup verifies and reuses completed assets. Incomplete transfers remain separate from ready files.

Other species use Ensembl reference retrieval during the relevant workflow. Splice prediction requires a compatible species model; [model configuration](llm-providers.md) describes language-model settings separately.

## Import existing references

```bash
modal-aso assets import --species human /path/to/existing/reference-directory
```

The directory should contain the matching FASTA, GTF, chromosome sizes, conservation tracks and RefSeq mapping files. Run `modal-aso assets list --species human` to see the expected filenames.

Import verifies the assembly and annotation, then links files into the managed data directory. Existing Bowtie indexes are checked against the genome before reuse. A missing index is built once.

The source directory can be read-only. It must remain accessible while the links are in use. Keep the managed data directory writable for indexes, locks and installation records.

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
