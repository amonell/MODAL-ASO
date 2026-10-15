# Preserve evaluation scope in supplied-ASO summaries

A parent run can evaluate the same molecule against a requested exon and again
against other local splice sites. Its summary previously searched every task's
score files, joined rows by molecule name or sequence, and filled missing fields
from that search. This could mix one evaluation's scores with another's skip
status, or assign evidence to the wrong repeated input row.

The summary now uses only the rows in its own evaluation payload. Input identity,
site names, zero scores and missing evidence survive unchanged. A missing score
remains missing even when another task scored the same molecule. Files from
sibling tasks remain available as separate evidence; their values do not fill
this task's summary fields.

Tests cover conflicting sibling scores/statuses, absent bound rows, repeated
names/sequences with distinct input identities, and the complete Markdown
summary. This change does not resolve inaccurate free-form prose or alter the
native scientific computations. Previously frozen runs and their first grades
are unchanged.
