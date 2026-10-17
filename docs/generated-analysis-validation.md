# Generated analysis execution and output integrity

Generated analysis programs are checked for statically unbound global names before
execution and after a proposed code repair. Missing imports or definitions enter
the existing repair loop with the original inputs and scientific criteria intact.
The validator does not choose an import or substitute a calculation. It respects
Python lexical scopes, aliases, closures and interpreter-provided script globals.
Programs using wildcard imports or dynamic namespaces retain runtime validation
because a lexical check cannot resolve their bindings reliably. This check does
not prove dependency installation, execution order, reachability or scientific
correctness. Syntax errors also enter the normal execution-repair path.

After an actual `NameError`, the runtime can identify a referenced global with
a later top-level function or class definition. Its value-free diagnostic
records the use, active module invocation and definition line numbers, and
explains that the definition must execute before the invocation. The existing
repair loop receives that context with the original program and traceback;
the host does not move definitions, alter calculations or add attempts. Valid
deferred references, missing definitions, external-module faults and local
binding errors are not labeled as this ordering defect. This is a narrow
explanation of an observed error, not a complete Python execution-order proof.

Each execution attempt must create its declared outputs. Before execution, any
existing result JSON, summary and explicitly requested output files are copied
to that attempt's `previous_outputs` directory. Their bytes and provenance are
flushed, their SHA-256 hashes are checked, and only then are the old output paths
removed. Receipts identify every archive, including after an unsuccessful
attempt. New files with identical correct contents are permitted. This avoids
using filesystem modification times to infer freshness, which is unreliable on
some network mounts. The archives are retained for inspection and recovery;
this is not a claim of transactional recovery across remote-server failure.

Conventional same-basename exports are promoted only when their content changed
during the attempt. Those paths may be inputs, so they are never removed merely
to test freshness. If the content is unchanged, the program must write to the
explicitly requested destination. Generated output paths cannot overlap the
declared read-only input files. Existing input restoration, tabular consistency,
requested-output and scientific result validation remain active.

Qualification covers missing and valid imports, nested scopes, dynamic namespace
fallback, syntax-error recovery, repair validation, stale outputs, timestamp-only
changes, identical regeneration, archive provenance and protected inputs. The
affected execution and repair suite passed 66 checks on a CIFS mount (one
unsupported symlink case excluded); the broader local-filesystem integration
suite passed 95 checks including that symlink case. These are software checks,
not evidence of ASO efficacy or a comparative benchmark win rate.
