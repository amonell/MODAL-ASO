"""Validate release identities without installing runtime dependencies."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import tomllib


def release_notes(root: Path, version: str) -> str:
    lines = (root / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line == f"## {version}" or line.startswith(f"## {version} — "):
            end = next((i for i in range(index + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
            notes = "\n".join(lines[index + 1:end]).strip()
            if notes:
                return notes + "\n"
    raise ValueError(f"CHANGELOG.md needs release notes for {version}")


def check_version(root: Path, tag: str | None = None) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    parts = version.split(".")
    if len(parts) != 3 or any(not p.isascii() or not p.isdecimal() or str(int(p)) != p for p in parts):
        raise ValueError("Stable releases use MAJOR.MINOR.PATCH, for example 1.0.0")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match package version v{version}")
    module = ast.parse((root / "aso_agents/__init__.py").read_text(encoding="utf-8"))
    runtime = next(ast.literal_eval(node.value) for node in module.body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    locked = [package["version"] for package in lock["package"]
              if package["name"] == project["name"] and package.get("source") == {"editable": "."}]
    citation = [line.partition(":")[2].strip().strip('"\'')
                for line in (root / "CITATION.cff").read_text(encoding="utf-8").splitlines()
                if line.startswith("version:")]
    for name, values in (("aso_agents.__version__", [runtime]), ("uv.lock", locked), ("CITATION.cff", citation)):
        if values != [version]:
            raise ValueError(f"{name} records {values!r}; expected {version}")
    release_notes(root, version)
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag", help="Expected Git tag, such as v1.0.0")
    parser.add_argument("--notes-file", type=Path, help="Write this version's changelog entry")
    args = parser.parse_args()
    version = check_version(args.root, args.tag)
    if args.notes_file:
        args.notes_file.write_text(release_notes(args.root, version), encoding="utf-8")
    print(f"Release metadata agrees: MODAL-ASO {version}")


if __name__ == "__main__":
    main()
