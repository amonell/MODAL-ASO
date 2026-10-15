# Program feature boundaries

A design can require the complete binding interval to remain in an annotated
UTR, coding region or exon while overlapping a shorter nominated site. These
are separate predicates. Overlap with the short site does not replace the
program's reference-feature boundary, and the complete ASO need not fit inside
the short site.

Steric and gapmer region compilation retain a boundary predicate supplied by
the program's region builder. When explicit site filters are present, the
compiled `tile_constraint_filters` append that predicate as an intersection
after the original union, intersection and exclusion expression. The same
predicate is recorded as `base_region_filter`. This preserves the boundary
when a pooled or rescored candidate table is filtered back into each program.
The source constraint count continues to count the original site constraints.

The boundary uses the original reference-feature rows and their coordinate
space. It does not substitute the intersection of those rows and the short
site as a new containment window. If the builder supplies no boundary policy
or no independent base region, no additional requirement is inferred from
labels or biological context. Mature-RNA steric compilation already retains
its independent transcript search bounds and continues to use that path.

Canonical `include` and `exclude` operators also survive normalization.
Exclusion is evaluated against each complete candidate and its stated overlap
threshold. It is not implemented by removing feature bases from the search
interval, which could remove candidates with a permitted partial overlap.
A leading exclusion searches a resolved independent base region. If that
domain is missing, compilation reports an unresolved scope instead of using
the excluded interval as the allowed region. An exclusion with no resolved
intervals in the output coordinate space also reports an error; missing
evidence is not interpreted as an empty forbidden region.

Explicit transcript rows are projected through the transcript named by each
row and its verified exon structure. A primary transcript is not a fallback
for a different accession or an unresolved version. This applies to rows
supplied as arrays as well as interval dictionaries.

Neutral tests cover both strands, genomic steric and transcript gapmer
coordinates, UTR and coding boundaries, alternative sites, exclusions and
constraint-only scopes. They validate interval filtering, not biological
efficacy or a benchmark win rate.
