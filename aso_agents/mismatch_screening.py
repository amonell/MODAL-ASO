"""Ungapped substitution evidence shared by native gene screening and exports."""
from collections import defaultdict

GENOMIC_TRANSCRIPT_STRAND_POLICY = (
    "Bowtie searches both genomic strands; the SAM alignment strand must match "
    "the GTF transcript strand because the query is transcript-oriented target RNA "
    "in the DNA alphabet. Opposite-strand exon overlaps are not counted; unresolved "
    "strand assignments make the count incomplete."
)


class MismatchQueryIndex:
    """Pigeonhole seeds prune candidates; full Hamming distance verifies each hit."""

    def __init__(self, queries, budget):
        if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 3:
            raise ValueError("Substitution screening supports integer budgets from 0 to 3")
        self.budget = budget
        self.queries = frozenset(queries)
        self.by_length = defaultdict(set)
        self.seeds = defaultdict(set)
        self.parts = {}
        for query in self.queries:
            if not query or set(query) - set("ACGT"):
                raise ValueError("Substitution queries must contain only A/C/G/T")
            length = len(query)
            self.by_length[length].add(query)
            if length <= budget:
                continue
            parts = [(i * length // (budget + 1), (i + 1) * length // (budget + 1))
                     for i in range(budget + 1)]
            self.parts[length] = parts
            for i, (start, end) in enumerate(parts):
                self.seeds[(length, i, query[start:end])].add(query)

    def matches(self, target):
        if set(target) - set("ACGT"):
            return []
        length = len(target)
        if length <= self.budget:
            candidates = self.by_length.get(length, ())
        else:
            candidates = set()
            for i, (start, end) in enumerate(self.parts.get(length, ())):
                candidates.update(self.seeds.get((length, i, target[start:end]), ()))
        return [(query, distance) for query in sorted(candidates)
                if (distance := sum(a != b for a, b in zip(query, target))) <= self.budget]


def empty_hits(budget=0):
    hit = {"transcripts": set(), "gene_ids": set(), "gene_names": set()}
    if budget:
        hit["by_mismatch"] = {str(mm): {"transcripts": set(), "gene_ids": set(), "gene_names": set()}
                              for mm in range(budget + 1)}
    return hit


def add_hit(hit, transcript, gene_id, gene_name, mismatches=0):
    buckets = [hit]
    if "by_mismatch" in hit:
        buckets.append(hit["by_mismatch"][str(mismatches)])
    for bucket in buckets:
        bucket["transcripts"].add(transcript)
        bucket["gene_ids"].add(gene_id)
        if gene_name:
            bucket["gene_names"].add(gene_name)


def sam_substitution_count(fields, budget):
    """Never turn a missing/invalid NM tag into an exact hit."""
    values = [field[5:] for field in fields[11:] if field.startswith("NM:i:")]
    if len(values) != 1:
        return None
    try:
        value = int(values[0])
    except ValueError:
        return None
    return value if 0 <= value <= budget else None


def annotate_mismatch_bins(row, budget, incomplete=False):
    """Retain observed identities, but only complete searches produce counts."""
    observed = row.get("off_target_observed_gene_ids_by_mismatch")
    names = row.get("off_target_observed_genes_by_mismatch")
    available = (isinstance(observed, dict) and isinstance(names, dict)
                 and all(isinstance(observed.get(f"{mm}mm"), list)
                         and isinstance(names.get(f"{mm}mm"), list) for mm in range(budget + 1)))
    complete = available and not incomplete
    ids_by_bin = {f"{mm}mm": sorted(set(observed[f"{mm}mm"])) if available and mm <= budget else None
                  for mm in range(4)}
    names_by_bin = {f"{mm}mm": sorted(set(names[f"{mm}mm"])) if available and mm <= budget else None
                    for mm in range(4)}
    row["Off_target_gene_ids_by_mismatch"] = ids_by_bin
    row["Off_target_genes_by_mismatch"] = names_by_bin
    union = set().union(*(ids_by_bin[f"{mm}mm"] for mm in range(budget + 1))) if available else None
    row["off_target_observed_gene_count"] = len(union) if union is not None else None
    row["off_target_count_status"] = ("complete_within_declared_scope" if complete else
                                    "incomplete" if incomplete else "unavailable")
    for mm in range(4):
        label = "exact" if mm == 0 else f"{mm}mm"
        row[f"Off_target_{label}_count"] = len(ids_by_bin[f"{mm}mm"]) if complete and mm <= budget else None
        row[f"Off_target_{mm}mm_gene_ids"] = ids_by_bin[f"{mm}mm"]
        row[f"Off_target_{mm}mm_genes"] = names_by_bin[f"{mm}mm"]
    total = len(union) if complete else None
    row["off_target_gene_count"] = total
    row["Times_mapped_off_target_genes"] = total
    row["Off_target_total_upto_mismatch_count"] = total
    row["Off_target_mismatch_budget"] = budget
    row["off_target_exact_count_unit"] = "genes"
    row["Off_target_weighted_risk_score"] = (
        sum(len(ids_by_bin[f"{mm}mm"]) / (mm + 1) for mm in range(budget + 1)) if complete else None)

