# Analysis over available reference data

The host previously rejected every sandbox analysis that combined submitted
molecules with target metadata. This forced an additional native evaluation even
when the complete finite reference and coordinate maps were already supplied.

The shared capability contract now permits sequence matching, orientation checks
and coordinate projection over available inputs. Independent graph review still
checks whether those inputs support every requested calculation. Target metadata
does not prove input completeness, and missing references must not be invented.

Mandatory external retrieval still requires a native operation. Biological
prediction, reference identity attestation, producer dependencies, native task
flags, actual ASO-set comparisons and final delivery retain their existing checks.
This removes one routing restriction without adding a task, model call or
gene-specific exception. Software qualification alone does not establish better
scientific accuracy or runtime on future requests.

## Preserve the requested operation

Reference availability does not change what the user asked the system to do.
For example, supplying an exon interval and asking for ASO design toward a
splicing outcome retains a native `design_new_asos` operation. Tiling is a method
within that design, not permission to replace the design with sequence arithmetic.
An explicitly scoped request to enumerate sequences or compute supplied rules
still belongs in `sandbox_analysis`, even if it mentions a biological motivation.
Missing design inputs follow the existing clarification and resolution paths.

Both existing graph-review passes now require a cited scope judgment for each
proposed task. The reviewer identifies the operation required by the original
evidence independently of the proposed type, and records whether scope is
preserved, reduced, expanded, or unsupported. The host converts discrepancies
into the existing graph-repair protocol; a top-level `valid` verdict cannot
override a discrepant check. Missing operations and data-flow edges remain the
responsibility of the independent operation-coverage pass. This applies to every
canonical task type and mixed graphs, without gene- or request-keyword routing.
Rejecting a no-op topology issue during adjudication does not skip that coverage
pass; both reviews must accept the plan before synthesis or execution proceeds.

The checks reuse the existing model calls and revision budgets. They do not
change scientific stage implementations, filters, scores, defaults, persisted
goal schemas, or legacy run loading. Native completion checks continue to apply
to verified native design tasks. A recorded `execution_limitation` is displayed
by the CLI alongside the available output paths instead of an unconditional
completion claim. Scope interpretation still depends on model judgments;
offline contract tests do not establish error-free classification of new prose.
