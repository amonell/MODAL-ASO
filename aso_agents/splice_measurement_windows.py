"""Keep nearby splice junctions independent when measuring model outputs."""
from copy import deepcopy
from math import isfinite, sqrt


def separate_measurement_windows(sites):
    """Limit each radius to less than half the nearest same-kind site distance.

    Isolated sites retain their requested radius. Duplicate representations of
    one junction do not count as neighboring junctions. Context for the model's
    input sequence is independent of these output measurement windows.
    """
    result = deepcopy(sites)
    groups = {}
    for site in result:
        for kind in ("acceptor", "donor"):
            point = site.get(kind)
            if not isinstance(point, dict) or type(point.get("position")) is not int:
                continue
            key = (str(point.get("chrom") or "").removeprefix("chr"), point.get("strand"), kind)
            groups.setdefault(key, {}).setdefault(point["position"], []).append(point)
    for positions in groups.values():
        ordered = sorted(positions)
        for index, position in enumerate(ordered):
            neighbors = ordered[max(0, index - 1):index] + ordered[index + 1:index + 2]
            if not neighbors:
                continue
            limit = (min(abs(position - other) for other in neighbors) - 1) // 2
            for point in positions[position]:
                radius = point.get("window_bp")
                if type(radius) is int and radius > limit:
                    point.update(requested_window_bp=radius, window_bp=limit,
                                 start=max(1, position - limit), end=position + limit,
                                 measurement_window_policy="separate_adjacent_junctions")
    for site in result:
        point = site.get(site.get("site_kind"))
        if isinstance(point, dict) and "window_bp" in point:
            site["window_bp"] = point["window_bp"]
    return result


def delta_metrics(pred_ref, pred_alt, acc_idx, don_idx, window, donor_window=None):
    """Measure acceptor and donor scores within their own declared radii."""
    result = {"pair": None, "acceptor": None, "donor": None}
    if pred_ref is None or pred_alt is None:
        return result
    values = {}
    for kind, index, radius, column in (
        ("acceptor", acc_idx, window, 1),
        ("donor", don_idx, window if donor_window is None else donor_window, 2),
    ):
        if not (0 <= index < pred_ref.shape[0] and 0 <= index < pred_alt.shape[0]):
            continue
        radius = max(0, int(radius))
        scores = [float(array[max(0, index - radius):min(array.shape[0], index + radius + 1), column].max())
                  for array in (pred_ref, pred_alt)]
        if all(isfinite(score) for score in scores):
            values[kind] = scores
            result[kind] = scores[1] - scores[0]
    if "acceptor" in values and "donor" in values:
        acc, don = values["acceptor"], values["donor"]
        result["pair"] = sqrt(max(acc[1], 0) * max(don[1], 0)) - sqrt(max(acc[0], 0) * max(don[0], 0))
    return result
