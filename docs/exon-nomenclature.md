# Exon names through design and reporting

MODAL-ASO resolves public exon names after selecting the reference transcript and
before planning and candidate review. The resolver uses the NCBI RefSeqGene/LRG
index and the corresponding EBI LRG XML record. Verification binds names to the
selected transcript version, assembly, strand and exact exon coordinates.

The run stores the source URLs, checksums, reference identity and mapping in its
selected-reference metadata. Public reference sequences remain outside the
report. Users need no separately downloaded mapping cache. Report rendering and
export read the saved evidence offline.

Resuming directly at candidate review also prepares any missing nomenclature.
Final artifact publication reloads the enriched goal so its source evidence
survives subsequent saving.

The original requested label stays in the goal. Resolved coordinates and the
transcript ordinal are stored separately. Candidate review receives the same
display names used by the binding plot, together with exact exon overlap for each
ASO. Junction-spanning candidates have separate overlap measurements for their
binding segments. Exon-focused proximity uses verified biological labels or an
explicitly declared `exon_index`.

A public record can be absent, unavailable, incompatible with the selected
version, or ambiguous about exon naming. When a requested exon defines design,
evaluation, binding constraints or recommendations, the run asks for its physical
identity before proceeding. This also applies to resumed runs. Repeated mentions
of the same unresolved feature produce one question.

The user can provide its complete genomic interval and assembly, or explicitly
identify its transcript-order exon in a specified reference. The compiler keeps
the original label and records the answer separately. Exact supplied coordinates
and verified transcript-conversion geometry can already establish the physical
feature, so these requests continue without another naming question. Execution
identities are bound to the reference structure and carried into tiling and
selection independently of public naming evidence.

Ordinary browsing and offline export can display transcript structure using
wording such as “ninth annotated exon.” A gene-wide request has no exon identity
question unless an exon-specific choice affects its execution. The source
failure itself removes no ASOs. Partial public transcripts remain unknown when
classifying exon-defined groups; users need not annotate these records manually.

Exon naming establishes feature identity. For an exon-defined target group,
exact exon spans identify included members and observed bypass junctions identify
excluded members. Partial or incompatible structures remain unknown. Membership
is computed once per sequence-backed catalog and reused across candidate scoring.
Target-group coverage records the fraction of confirmed members matched;
selectivity records preference for the target set, including recorded penalties
for protected and unresolved matches. A single accession supplies the coordinate
reference. Target, protected, other and unresolved matches retain distinct counts.
Saved historical ranks and single-transcript scores retain their original values.
