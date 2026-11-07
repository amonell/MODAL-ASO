# Reference scope and splice planning reports

An exon-window request could produce valid candidates yet fail the delivery
minimum check because its compiled splice window had no explicit interval
constraint count. The minimum check now reconstructs the window from the
recorded exon boundaries and flank parameters. It accepts the scope only when
the physical rows match exactly and no user constraints remain unresolved.
This does not establish an ASO's biological effect.

For explicitly transcript-relative intervals, a chromosome label in the interval
reference slot can now be bound to the declared transcript. Binding requires a
unique, version-consistent transcript and the same reference chromosome. Bounds
and coordinate conventions are preserved and subsequently validated; the
original reference and the binding evidence remain in the materialized record.
An ambiguous transcript, conflicting chromosome or out-of-bounds interval still
fails. This identity repair does not resolve contradictory requested constraints.

Splice strategy reports now render the structured planning priorities and
reference catalog directly. They retain the desired directions and safeguards,
show shared and unresolved references, and distinguish planning choices from
measured effects. Unverified free-form mechanistic assertions no longer become
report facts merely because the planning model produced them. Direct masking of
a canonical splice junction does not establish increased exon inclusion; see
[Hua et al., 2007](https://doi.org/10.1371/journal.pbio.0050073).

The packaged tiler's direct scoring script also locates the enclosing application
correctly when importing the shared oligonucleotide chemistry encoder.

Automatic safeguards recognize the acceptor and donor of the targeted exon as
two boundaries of the same requested outcome. For inclusion, the paired boundary
may increase and a directional safeguard checks decreases. For skipping, it may
decrease and the safeguard checks increases. The default opposing-change limit
remains 0.05 in model-score units.

This relationship is checked against the current job's physical splice plan and
the selected transcript's exact exon coordinates, chromosome and strand. A
paired boundary outside the ordinary neighborhood radius is added to measurement
context. Ambiguous or conflicting reference geometry retains the existing policy.
Explicit bounds, required guards, preserved transcripts and conflicting isoform
objectives retain their constraints. Other exons keep their existing safeguards.
The relationship is recomputed for each job and recorded in the safeguard
selection audit, including the physical boundaries and reference transcript.

Validation covers both strands and window contexts, altered and unresolved
reference data, transcript namespace ambiguity and bounds, structured report
rendering, and the actual Snakemake scoring command. Saved-input checks are
offline diagnostics; they do not replace first benchmark outcomes or demonstrate
end-to-end success on a new run.
