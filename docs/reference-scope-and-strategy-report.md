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

The configured model reviews every proposed automatic splice safeguard for the
current design job. It classifies increases and decreases at each site as intended,
acceptable, unwanted or uncertain, gives a reason, and chooses a two-sided check,
a directional check, or observation without a ranking penalty. Explicit user
bounds and required preservation checks are included as locked policies.

Code supplies checked exon geometry, transcript objectives, reference scope and
the original request. The acceptor and donor of the targeted exon form a paired
outcome: inclusion can favor increases at both boundaries, and skipping can favor
decreases at both. Alternative-site and intron events receive their own objective
context. Numeric execution limits remain the recorded user bounds or the default
0.05 in model-score units. The reviewer chooses their biological relevance and
direction; it cannot introduce new numeric limits.

This relationship is checked against the current job's physical splice plan and
the selected transcript's exact exon coordinates, chromosome and strand. A
paired boundary outside the ordinary neighborhood radius is added to measurement
context. Ambiguous or conflicting reference geometry retains the existing policy.
Explicit bounds and required preservation guards retain their constraints. Code
validates case identities, directional consistency, exact coverage and locked
policies before applying decisions. Observations retain their measurements and
avoid duplicate filtering by older shared-site summaries. Reviews are recomputed
for each job. The selection audit records the context, complete model responses,
validation outcomes, decisions and reasons.

Reviews use bounded batches with one repair attempt for incomplete or invalid
decisions. Successful decisions survive an outage in a later batch. Unreviewed
cases retain the recorded default policy, and the audit and report identify the
incomplete review. This fallback lets the run finish with its evidence visible.

Validation covers both strands and window contexts, altered and unresolved
reference data, transcript namespace ambiguity and bounds, structured report
rendering, and the actual Snakemake scoring command. Saved-input checks are
offline diagnostics; they do not replace first benchmark outcomes or demonstrate
end-to-end success on a new run.
