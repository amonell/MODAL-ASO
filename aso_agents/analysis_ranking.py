"""Rank caller-supplied comparable keys without inventing selection criteria."""
from decimal import Decimal
import math
from numbers import Rational, Real


def _validate_key(value):
    if isinstance(value, tuple):
        for component in value:
            _validate_key(component)
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Dense ranking requires known finite key components")
    elif isinstance(value, Real):
        if not isinstance(value, Rational) and not math.isfinite(value):
            raise ValueError("Dense ranking requires known finite key components")
    elif not isinstance(value, str):
        raise TypeError("Dense ranking keys must be known numbers, strings, or tuples of them; "
                        "keep missing-key rows unranked unless the user specifies a policy")


def dense_ranks(keys, *, reverse=False):
    """Return one-based, gap-free ranks in the original input order.

    Equal complete keys share a rank; the next distinct key advances by one.
    Keys are compared as supplied, including exact Decimal/Fraction arithmetic;
    no coercion, rounding, identifier tie-break, filtering or selection occurs.
    Tuples use Python lexicographic order. reverse changes the whole key order;
    the caller must construct keys for any mixed ascending/descending priorities.
    Missing or nonfinite components are rejected, never assigned an invented rank.
    """
    if type(reverse) is not bool:
        raise TypeError("reverse must be a Boolean")
    values = list(keys)
    for value in values:
        _validate_key(value)
    try:
        order = sorted(range(len(values)), key=values.__getitem__, reverse=reverse)
    except TypeError as exc:
        raise TypeError("Dense ranking requires mutually comparable complete keys") from exc
    ranks = [0] * len(values)
    rank = 0
    previous = None
    for index in order:
        value = values[index]
        if rank == 0 or value != previous:
            rank += 1
        ranks[index] = rank
        previous = value
    return ranks
