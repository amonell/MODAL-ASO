"""Explicit coordinates for one contiguous sequence, without inferred origins.

The first character has ``first_coordinate``; coordinates change by ``direction``
per character. This represents local offsets or a contiguous genomic interval,
not a spliced transcript projection, alignment with gaps or assembly liftover.
"""
from dataclasses import dataclass
from numbers import Integral


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a Boolean or inferred coordinate")
    return int(value)


@dataclass(frozen=True, kw_only=True)
class SequenceCoordinates:
    first_coordinate: int
    length: int
    direction: int = 1

    def __post_init__(self):
        for name in ("first_coordinate", "length", "direction"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.length <= 0:
            raise ValueError("Sequence length must be positive")
        if self.direction not in (-1, 1):
            raise ValueError("Coordinate direction must be +1 or -1")

    @classmethod
    def from_inclusive_bounds(cls, *, lower: int, upper: int, direction: int):
        """Build a frame from independently declared ascending source bounds.

        ``direction`` describes the supplied string's coordinate order, not an
        inferred biological strand. The expected length comes from the bounds;
        ``extract`` then checks the actual sequence against that independent
        length. Matching lengths do not establish sequence identity or provenance.
        """
        lower = _integer(lower, "lower")
        upper = _integer(upper, "upper")
        direction = _integer(direction, "direction")
        if lower > upper:
            raise ValueError("Inclusive bounds must be in ascending coordinate order")
        return cls(first_coordinate=lower if direction == 1 else upper,
                   length=upper - lower + 1, direction=direction)

    def to_index(self, coordinate: int) -> int:
        """Convert a base coordinate to its bounded zero-based string index."""
        coordinate = _integer(coordinate, "coordinate")
        index = (coordinate - self.first_coordinate) * self.direction
        if not 0 <= index < self.length:
            raise ValueError(f"Coordinate {coordinate} is outside this sequence frame")
        return index

    def to_coordinate(self, index: int) -> int:
        """Convert a bounded zero-based string index to its base coordinate."""
        index = _integer(index, "index")
        if not 0 <= index < self.length:
            raise ValueError(f"Index {index} is outside sequence length {self.length}")
        return self.first_coordinate + self.direction * index

    def inclusive_slice(self, lower: int, upper: int) -> slice:
        """Ascending inclusive coordinate bounds to a slice in sequence order.

        Both endpoints must exist. No clipping, reverse complementation or
        coordinate-origin inference occurs, including for decreasing frames.
        """
        lower = _integer(lower, "lower")
        upper = _integer(upper, "upper")
        if lower > upper:
            raise ValueError("Inclusive bounds must be in ascending coordinate order")
        indices = (self.to_index(lower), self.to_index(upper))
        return slice(min(indices), max(indices) + 1)

    def extract(self, sequence: str, lower: int, upper: int) -> str:
        """Extract the entire inclusive interval in the supplied orientation."""
        if not isinstance(sequence, str):
            raise TypeError("Sequence must be a string")
        if len(sequence) != self.length:
            raise ValueError(f"Sequence length {len(sequence)} differs from frame length {self.length}")
        return sequence[self.inclusive_slice(lower, upper)]
