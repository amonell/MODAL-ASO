"""Host-observed envelope changes for the existing post-repair review."""

from copy import deepcopy
from typing import Any, Dict, List


def envelope_changes(before: Any, after: Any) -> List[Dict[str, Any]]:
    """Compare actual snapshots, including normalization, without inferring intent.

    Resized arrays are shown whole: a removed member must not be described as
    changing into its former neighbor. Equal-length arrays use positional paths.
    Presence is explicit so missing, null, false and zero remain distinguishable.
    """
    changes: List[Dict[str, Any]] = []

    def state(present: bool, value: Any) -> Dict[str, Any]:
        return {"present": True, "value": deepcopy(value)} if present else {"present": False}

    def visit(old: Any, new: Any, path: str, old_present: bool = True,
              new_present: bool = True) -> None:
        if old_present and new_present:
            if isinstance(old, dict) and isinstance(new, dict):
                for key in sorted(old.keys() | new.keys()):
                    token = str(key).replace("~", "~0").replace("/", "~1")
                    visit(old.get(key), new.get(key), path + "/" + token,
                          key in old, key in new)
                return
            if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
                for index, (old_item, new_item) in enumerate(zip(old, new)):
                    visit(old_item, new_item, path + "/" + str(index))
                return
            if type(old) is type(new) and old == new:
                return
        changes.append({"path": path, "before": state(old_present, old),
                        "after": state(new_present, new)})

    visit(before, after, "")
    return changes
