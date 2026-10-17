"""Explicit one-based inclusive projections for one complete spliced transcript.

This module performs coordinate arithmetic only. The caller must establish the
transcript, assembly and reference exon identity; labels cannot verify them.
"""
from dataclasses import dataclass

from .sequence_coordinates import _integer


@dataclass(frozen=True, kw_only=True)
class SplicedSequenceCoordinates:
    chromosome: str
    strand: str
    exons: tuple

    def __post_init__(self):
        if not isinstance(self.chromosome, str) or not self.chromosome.strip():
            raise ValueError("A known chromosome is required; no default is inferred")
        if self.strand not in ("+", "-"):
            raise ValueError("Strand must be explicitly '+' or '-'")
        intervals = []
        for exon in self.exons:
            if not isinstance(exon, (tuple, list)) or len(exon) != 2:
                raise ValueError("Each exon must be a genomic (start1, end1) pair")
            start, end = (_integer(v, "exon endpoint") for v in exon)
            if not 1 <= start <= end:
                raise ValueError("Exon bounds must be positive ascending inclusive coordinates")
            intervals.append((start, end))
        if not intervals:
            raise ValueError("A complete nonempty exon list is required")
        intervals.sort()
        if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("One transcript's exons must not overlap or duplicate")
        if self.strand == "-":
            intervals.reverse()
        object.__setattr__(self, "exons", tuple(intervals))

    @property
    def length(self):
        return sum(end - start + 1 for start, end in self.exons)

    def project_interval(self, start1, end1):
        """Return genomic blocks in mature 5prime-to-3prime order, never a span.

        start1 and end1 are mature/cDNA positions in 1..length. Both input and
        output endpoints are one-based inclusive. Convert an exonic genomic
        position with to_mature(genomic_position1) before projecting it.
        Genomic bounds
        within each block are ascending on either strand; block order follows
        the transcript and is therefore decreasing genomically on minus strand.
        The complete exon list defines mature position one; no clipping occurs.
        """
        start1, end1 = _integer(start1, "start1"), _integer(end1, "end1")
        if not 1 <= start1 <= end1 <= self.length:
            raise ValueError(
                "Interval must lie entirely within the mature sequence: "
                f"project_interval received start1={start1}, end1={end1}; "
                f"expected mature/cDNA coordinates with 1 <= start1 <= end1 <= {self.length}. "
                "Convert an exonic genomic position with to_mature(genomic_position1). "
                "No coordinates were converted or clipped."
            )
        blocks, cursor = [], 1
        for ordinal, (left, right) in enumerate(self.exons, 1):
            exon_end = cursor + right - left
            a, b = max(start1, cursor), min(end1, exon_end)
            if a <= b:
                lo, hi = ((left + a - cursor, left + b - cursor) if self.strand == "+"
                          else (right - (b - cursor), right - (a - cursor)))
                blocks.append({"chrom": self.chromosome, "strand": self.strand,
                               "genomic_start1": lo, "genomic_end1": hi,
                               "mature_start1": a, "mature_end1": b,
                               "exon_ordinal": ordinal})
            cursor = exon_end + 1
        return blocks

    def to_genomic(self, mature_position1):
        """Map one mature base to its chromosome coordinate."""
        return self.project_interval(mature_position1, mature_position1)[0]["genomic_start1"]

    def to_mature(self, genomic_position1):
        """Map an exonic base; intronic and out-of-reference bases are errors."""
        position = _integer(genomic_position1, "genomic_position1")
        cursor = 1
        for left, right in self.exons:
            if left <= position <= right:
                return cursor + (position - left if self.strand == "+" else right - position)
            cursor += right - left + 1
        raise ValueError(
            "Genomic position is not an exonic base of this transcript: "
            f"to_mature received genomic_position1={position} on {self.chromosome} "
            f"({self.strand} strand). It requires a one-based genomic exon position. "
            f"A mature/cDNA position in 1..{self.length} maps through to_genomic(mature_position1). "
            "Intronic and outside positions are not projected."
        )

    def extract(self, start1, end1, *, fetch_genomic):
        """Fetch plus-reference blocks and return transcript-oriented sequence.

        start1 and end1 are mature/cDNA positions in 1..length.
        fetch_genomic(chrom, start1, end1) must return exactly the ascending,
        one-based inclusive genomic interval as a string. Minus-strand blocks
        are reverse-complemented individually and joined in transcript order.
        Compare the result to the independent mature sequence when verifying
        reference correspondence; this method alone makes no agreement claim.
        """
        pieces = []
        complement = str.maketrans("ACGTRYMKBDHVSWN", "TGCAYRKMVHDBSWN")
        for block in self.project_interval(start1, end1):
            lo, hi = block["genomic_start1"], block["genomic_end1"]
            sequence = fetch_genomic(self.chromosome, lo, hi)
            if not isinstance(sequence, str) or len(sequence) != hi - lo + 1:
                raise ValueError("Fetched genomic sequence must have the exact block length")
            sequence = sequence.upper()
            if set(sequence) - set("ACGTRYMKBDHVSWN"):
                raise ValueError("Fetched genomic sequence must use the IUPAC DNA alphabet")
            pieces.append(sequence if self.strand == "+" else sequence.translate(complement)[::-1])
        return "".join(pieces)
