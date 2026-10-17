"""Classify integer revisions against their exact, caller-defined registry keys."""
from .analysis_ranking import _validate_key
from .sequence_coordinates import _integer


def _scope(value):
    if not isinstance(value, tuple) or not value:
        raise TypeError("Scope must be a nonempty tuple containing every identity field")
    _validate_key(value)
    hash(value)
    return value


class RegisteredVersions:
    """Index (scope_tuple, integer_version) pairs from the complete registry.

    A scope tuple must include every requested grouping field. Versions are
    compared numerically and the greatest *registered* version is selected per
    scope, even if that version has no observations. Status/coverage metadata
    does not alter selection. The caller handles any different selection rule.
    Duplicate identical registry keys do not change membership or selection.
    """

    def __init__(self, registrations):
        self._registered = set()
        self._latest = {}
        for scope, version in registrations:
            scope, version = _scope(scope), _integer(version, "registered version")
            self._registered.add((scope, version))
            if scope not in self._latest or version > self._latest[scope]:
                self._latest[scope] = version

    def latest(self, scope):
        """Return greatest registered version, or None for an unknown scope."""
        return self._latest.get(_scope(scope))

    def classify(self, scope, version):
        """Return current, superseded or unregistered for the exact complete key.

        A version smaller than the selected one is superseded only if its own
        complete scope/version key is registered. Absence from this registry
        establishes neither a failed search nor an absence of biological hits.
        """
        scope, version = _scope(scope), _integer(version, "observed version")
        if (scope, version) not in self._registered:
            return "unregistered"
        return "current" if version == self._latest[scope] else "superseded"
