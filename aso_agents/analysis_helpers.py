"""Publish shared analysis APIs and reject generated substitutes under their names.

This is a narrow binding check, not a scientific validator or security boundary.
Custom calculations under other names still require scientific review.
"""
import ast


_HELPERS = (
    ("RegisteredVersions", "aso_agents.registered_versions",
     "RegisteredVersions(registrations)",
     "registrations are (complete_scope_tuple, integer_version) pairs; latest(scope) "
     "returns greatest registered version or None; classify(scope,version) returns "
     "current/superseded/unregistered after exact membership checking; metadata and "
     "observation presence do not change latest selection; no coverage/count inference"),
    ("SplicedSequenceCoordinates", "aso_agents.spliced_coordinates",
     "SplicedSequenceCoordinates(*, chromosome, strand, exons)",
     "exons are complete genomic (start1,end1) pairs; length is mature transcript length; "
     "project_interval(start1,end1) takes mature/cDNA endpoints in 1..length and returns "
     "chrom,strand,genomic_start1,genomic_end1,mature_start1,mature_end1,exon_ordinal blocks "
     "in transcript order; to_genomic(mature_position1) maps one mature base to genomic; "
     "to_mature(genomic_position1) maps one genomic exon base to mature and rejects introns; "
     "extract(start1,end1,fetch_genomic=callable) also takes mature/cDNA endpoints, but its "
     "fetch_genomic(chrom,start1,end1) callback receives plus-genomic bounds; returns "
     "transcript-oriented DNA; all endpoints one-based inclusive; no origin/strand inference or clipping"),
    ("SequenceCoordinates", "aso_agents.sequence_coordinates",
     "SequenceCoordinates(*, first_coordinate, length, direction=1)",
     "to_index(coordinate), to_coordinate(index), inclusive_slice(lower,upper), "
     "extract(sequence,lower,upper); bounds are ascending inclusive; no implicit origin or clipping"),
    ("compare_antisense_target", "aso_agents.mapping_evidence",
     "compare_antisense_target(antisense_5to3, target_5to3)",
     "dict: mismatch_count, match_count, mismatch_positions_in_supplied_sequence_1based, "
     "mismatch_positions_in_target_sequence_1based, mismatch_details_in_target_orientation; "
     "equal-length canonical sequences only"),
    ("interval_overlap_summary", "aso_agents.region_overlap",
     "interval_overlap_summary(start,end,feature_start,feature_end,*,coordinate_convention)",
     "dict: overlap_interval, overlap_nt, any_overlap, contains_feature, within_feature; "
     "coordinate_convention must be inclusive or half_open"),
    ("summarize_required_criteria", "aso_agents.analysis_criteria",
     "summarize_required_criteria(criteria, *, required_names=None)",
     "dict: status (passed/failed/indeterminate), criteria, passed_criteria, failed_criteria, "
     "unverified_criteria; omitted required names stay unknown; extra names are rejected"),
    ("dense_ranks", "aso_agents.analysis_ranking",
     "dense_ranks(keys, *, reverse=False)",
     "list of one-based gap-free integer ranks in original row order; equal complete keys "
     "share a rank; numbers, strings and tuples compare as supplied without rounding; "
     "missing/nonfinite components rejected; no filtering or added tie-breakers"),
)


def shared_analysis_helpers() -> list[dict]:
    return [{"name": name, "module": module, "signature": signature, "returns": result}
            for name, module, signature, result in _HELPERS]


def validate_shared_helper_bindings(code: str) -> None:
    """Reserve the advertised bare API names for their explicit native imports.

    Module-qualified imports and aliases are allowed; this deliberately does not
    attempt to resolve arbitrary aliases or prove runtime call targets. Object
    attributes with the same names do not replace a bare helper binding.
    """
    modules = {name: module for name, module, _, _ in _HELPERS}
    errors = []

    class Bindings(ast.NodeVisitor):
        scope = "module"

        def check(self, name, node):
            if self.scope != "class" and name in modules:
                errors.append(f"line {node.lineno}: {name}; import it with "
                              f"'from {modules[name]} import {name}'")

        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                self.check(node.id, node)

        def visit_arg(self, node):
            self.check(node.arg, node)

        def visit_FunctionDef(self, node):
            self.check(node.name, node)
            previous, self.scope = self.scope, "function"
            self.generic_visit(node)
            self.scope = previous

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            self.check(node.name, node)
            previous, self.scope = self.scope, "class"
            self.generic_visit(node)
            self.scope = previous

        def visit_ImportFrom(self, node):
            for alias in node.names:
                name = alias.asname or alias.name
                if (name in modules and
                        (node.level or node.module != modules[name] or alias.name != name)):
                    self.check(name, node)

        def visit_Import(self, node):
            for alias in node.names:
                self.check(alias.asname or alias.name.split(".")[0], node)

    Bindings().visit(ast.parse(code))
    if errors:
        raise ValueError("Do not redefine a shared analysis helper: " + "; ".join(errors)
                         + ". Import the native API and use its documented return contract. "
                         "Keep distinct custom operations under distinct names.")
