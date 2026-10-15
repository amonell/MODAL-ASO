# Evaluation reference identity

A downstream evaluation can select a different organism or gene from the design
task that supplied its molecules. Previously, the evaluator mapped sequences using
the selected transcript's reference metadata, but could still pass the design's
organism and gene ID into off-target screening. The report could consequently
describe a reference from one organism beside screening results from another.

The evaluator now binds its screening context to the resolved primary transcript
before loading genomes or scoring molecules. Explicit transcript resolution
retains the reference gene ID as well as the organism, assembly and version.
Accession prefixes, gene-name casing and free-text descriptions do not determine
this binding.

Canonical task fields distinguish a declared identity from defaults introduced
by runtime normalization. A task that declares its own target identity can replace
inherited defaults with verified reference facts. An explicit task organism or
gene ID that contradicts the reference stops evaluation with a structured error.
When the canonical task is unavailable, the supplied evaluation envelope is
treated as authoritative and contradictions also stop evaluation.

On an identity change, the evaluator clears the old gene-resolution cache and
catalog-completeness marker, and attempts to load the new gene's transcript
catalog. An inherited patient reference is not applied to a different organism.
References outside the primary gene and organism remain available as separately
identified mapping evidence, but do not enter the primary isoform-scoring catalog.
Unattested catalog entries are excluded from that scoring scope after rebinding.
The context summary records the previous identity, the reference binding, cleared
fields and exact catalog labels used for screening. Missing reference metadata
remains explicitly attributed to the evaluation context rather than fabricated.

The regression tests exercise neutral transcript identifiers, different organisms
and genes, explicit conflicts, missing metadata, reference refresh and the complete
evaluation-to-scoring call. Tests also retain the existing per-reference mapping
and version-lock behavior. Saved-run qualification checks context binding against
the installed annotation without changing the original run, scores or reviews.
