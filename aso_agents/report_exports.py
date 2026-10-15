"""Portable copies of native reports and their explicitly linked evidence.

This module changes URL destinations, never report text or scientific results.
Only existing files inside the producing run are collected; missing references
are retained and recorded, without searching other runs for a matching name.
"""
from __future__ import annotations

import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote, urlsplit
import uuid


REPORT_SUFFIXES = {".md", ".markdown", ".html", ".htm"}
_ATTRIBUTE = re.compile(r'''([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?''')
_DESTINATION = r'''(?:<(?P<angle>[^<>\n]*)>|(?P<bare>(?:\\.|[^\s()\\]|\([^()\s]*\))*))'''
_INLINE_LINK = re.compile(
    r"(?<!\\)\[[^\]\n]*\]\(\s*" + _DESTINATION
    + r'''\s*(?:(?:"[^"\n]*"|'[^'\n]*'|\([^()\n]*\))\s*)?\)'''
)
_REFERENCE_LINK = re.compile(r"^ {0,3}\[[^\]\n]+\]:[ \t]*" + _DESTINATION, re.M)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _HTMLLinks(HTMLParser):
    """Locate original attribute values without serializing/reformatting HTML."""

    def __init__(self, text):
        super().__init__(convert_charrefs=False)
        self.offsets = [0]
        self.offsets.extend(match.end() for match in re.finditer("\n", text))
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        raw = self.get_starttag_text()
        line, column = self.getpos()
        offset = self.offsets[line - 1] + column
        for match in _ATTRIBUTE.finditer(raw):
            if match.group(1).lower() not in {"href", "src", "poster", "data"}:
                continue
            group = next((i for i in (2, 3, 4) if match.group(i) is not None), None)
            if group is None:
                continue
            if tag.lower() == "base":
                raise ValueError("native_report_export_html_base_unsupported")
            self.links.append((offset + match.start(group), offset + match.end(group),
                               html.unescape(match.group(group)), True))

    handle_startendtag = handle_starttag


def _markdown_links(text):
    # Mask code while keeping character offsets. Native reports use fenced and
    # inline code extensively for molecules, paths and literal source markup.
    chars = list(text)
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        protected = fence is not None or bool(marker) or line.startswith(("    ", "\t"))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not line[marker.end():].strip():
                fence = None
        elif marker:
            fence = marker[1]
        if protected:
            chars[offset:offset + len(line)] = ["\n" if c == "\n" else " " for c in line]
        offset += len(line)
    masked = "".join(chars)
    for match in re.finditer(r"(?<![\\`])(`+)(?!`)(.*?)\1(?!`)", masked, re.S):
        chars[match.start():match.end()] = ["\n" if c == "\n" else " " for c in match[0]]
    masked = "".join(chars)
    links = []
    for pattern in (_INLINE_LINK, _REFERENCE_LINK):
        for match in pattern.finditer(masked):
            group = "angle" if match.group("angle") is not None else "bare"
            start, end = match.span(group)
            value = html.unescape(re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])", r"\1", text[start:end]))
            links.append((start, end, value, False))
    return sorted(links)


def _links(text, suffix):
    return _HTMLLinks(text).links if suffix in {".html", ".htm"} else _markdown_links(text)


def _reference(source, root, value):
    try:
        url = urlsplit(value)
        if url.scheme not in {"", "file"} or url.netloc or not url.path:
            return None
        candidate = (source.parent / unquote(url.path)).resolve()
        if not candidate.is_relative_to(root):
            return {"reason": "outside_producing_run", "url": value}
        return {"path": candidate, "url": value,
                "reason": None if candidate.is_file() else "file_unavailable",
                "suffix": ("?" + url.query if url.query else "") + ("#" + url.fragment if url.fragment else "")}
    except (ValueError, OSError, RuntimeError):
        return {"reason": "invalid_local_reference", "url": value}


def _write_immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("native_report_export_integrity_error:" + str(path))
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_native_report(source, destination, *, source_root):
    """Package the reachable local report/evidence files under ``name.assets``.

    The bundle keeps the producing run's relative hierarchy, so cycles and
    repeated basenames are unambiguous. Its identity includes all original file
    hashes. Document links are rebased; every other byte and every data asset
    is preserved. Dynamic JavaScript/CSS references are outside this parser's
    scope; native HTML uses inline scripts/styles and explicit href/src links.
    """
    source_argument, destination_argument = str(source), str(destination)
    source, destination, root = Path(source).resolve(), Path(destination).resolve(), Path(source_root).resolve()
    if source == destination:
        raise ValueError("native_report_export_would_replace_source:" + str(source))
    if not source.is_relative_to(root):
        raise ValueError("native_report_export_source_outside_run:" + str(source))
    nodes, pending, unresolved = {}, [source], []
    while pending:
        path = pending.pop()
        if path in nodes:
            continue
        node = {"source_sha256": _sha(path), "links": []}
        nodes[path] = node
        if path.suffix.lower() not in REPORT_SUFFIXES:
            continue
        data = path.read_bytes()
        try:
            node["text"] = data.decode("utf-8")
        except UnicodeDecodeError:
            if path == source:
                raise ValueError("native_report_export_source_not_utf8:" + str(path))
            continue
        if hashlib.sha256(data).hexdigest() != node["source_sha256"]:
            raise ValueError("native_report_export_source_changed:" + str(path))
        for span in _links(node["text"], path.suffix.lower()):
            ref = _reference(path, root, span[2])
            if ref is None:
                continue
            node["links"].append((span, ref))
            if ref["reason"]:
                unresolved.append({"source": path.relative_to(root).as_posix(),
                                   "url": ref["url"], "reason": ref["reason"]})
            else:
                pending.append(ref["path"])
    identity = {path.relative_to(root).as_posix(): node["source_sha256"]
                for path, node in nodes.items()}
    digest = hashlib.sha256(json.dumps({"format": 1, "entry": source.relative_to(root).as_posix(),
                                       "sources": identity}, sort_keys=True).encode()).hexdigest()
    bundle = destination.with_name(destination.name + ".assets") / digest
    files = bundle / "files"

    def render(node, target):
        text = node["text"]
        for (start, end, _, is_html), ref in reversed(node["links"]):
            if "path" not in ref:
                continue
            copied = files / ref["path"].relative_to(root)
            href = quote(Path(os.path.relpath(copied, target.parent)).as_posix(), safe="/.") + ref["suffix"]
            text = text[:start] + (html.escape(href, quote=True) if is_html else href) + text[end:]
        return text.encode("utf-8")

    exported = {}
    for path, node in sorted(nodes.items()):
        relative = path.relative_to(root)
        target = files / relative
        if "text" in node:
            data = render(node, target)
            _write_immutable(target, data)
            result_sha = hashlib.sha256(data).hexdigest()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    shutil.copyfile(path, temporary)
                    if _sha(temporary) != node["source_sha256"]:
                        raise ValueError("native_report_export_source_changed:" + str(path))
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            result_sha = _sha(target)
            if result_sha != node["source_sha256"]:
                raise ValueError("native_report_export_integrity_error:" + str(target))
        exported[relative.as_posix()] = {"source_sha256": node["source_sha256"], "sha256": result_sha}
    for path, node in nodes.items():
        if _sha(path) != node["source_sha256"]:
            raise ValueError("native_report_export_source_changed:" + str(path))
    manifest = {"format": 1, "entry": source.relative_to(root).as_posix(),
                "files": exported, "unresolved_links": unresolved,
                "scope": "Explicit Markdown links and HTML URL attributes; no dynamic JavaScript or CSS URL parsing."}
    manifest_path = bundle / "manifest.json"
    _write_immutable(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    data = render(nodes[source], destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"source": source_argument, "destination": destination_argument, "source_root": str(root),
            "source_sha256": nodes[source]["source_sha256"], "sha256": hashlib.sha256(data).hexdigest(),
            "method": "native_report_bundle", "manifest": str(manifest_path),
            "manifest_sha256": _sha(manifest_path), "asset_count": len(nodes), "unresolved_links": unresolved}
