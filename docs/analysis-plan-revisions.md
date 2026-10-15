# Analysis plans and executed revisions

An analysis-code repair previously preserved every descriptive field from the
original plan. A repaired export filename could therefore disagree with the plan
and its expected-output list. A preexisting description of an unimplemented
conversion could also survive every code repair.

Native repairs now bind to both the failed program and the preceding plan. The
program binding covers the exact code string; the plan binding covers canonical
JSON with sorted keys. A repair includes explicit `plan_updates` for descriptive
fields that need correction: `analysis_plan`, `assumptions`, `limitations` and
`expected_outputs`. An empty list records that no descriptive update was proposed.
Unsupported fields, duplicate updates, incorrect value types and stale bindings
are rejected. Code edits retain their existing line-range and dependency checks.

The generator and repairer are instructed to reconcile these descriptions with
the complete program. A mismatch with a user requirement must be fixed in code;
editing a description does not excuse or override the requirement. Required
output paths remain in the execution contract and are verified independently of
the descriptive `expected_outputs` list.

Each attempt retains its original plan and source. Execution receipts now identify
those attempt-specific paths and include their file-content hashes. The final
manifest uses the accepted revision's descriptions. Previous attempts are not
rewritten. Historical code-only repair records remain supported by the low-level
replay helper; new native repair requests require both bindings and an explicit
update list.

These checks establish revision identity and restrict editable metadata. They do
not prove that the model's natural-language descriptions accurately summarize all
program behavior; that remains a scientific reporting concern for fresh execution
and independent review.
