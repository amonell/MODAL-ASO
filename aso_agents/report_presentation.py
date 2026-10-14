"""Offline presentation of saved evidence; no inference, selection or reranking."""
from __future__ import annotations

import html
import json
import os
import re
import hashlib
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


TABLE_CSS = """
.report-content{min-width:0;overflow-wrap:anywhere}
.report-content .table-scroll{max-width:100%;overflow-x:auto;margin:16px 0;border:1px solid #d7e4f5;border-radius:10px}
.report-content table{border-collapse:collapse;min-width:100%;width:max-content;margin:0}
.report-content th,.report-content td{padding:10px 12px;border-bottom:1px solid #d7e4f5;text-align:left;vertical-align:top;max-width:38rem}
.report-content th{background:#edf4ff;font-weight:600}
.report-content code{font-family:Consolas,monospace;overflow-wrap:anywhere;background:#f5f7fa;border-radius:4px;padding:1px 4px}
.report-content pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:100%;background:#f5f7fa;padding:16px;border-radius:10px}
.report-content a{color:#225fb7}.report-content p,.report-content li{line-height:1.6}
.report-content .sequence{font-family:Consolas,monospace;white-space:nowrap}
.report-content .downloads{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.report-content details{margin:12px 0}.report-content summary{cursor:pointer;font-weight:600}
.report-content :focus-visible{outline:2px solid #225fb7;outline-offset:3px}
@media(max-width:600px){.report-content th,.report-content td{padding:8px;font-size:14px}.report-content h1{font-size:1.5rem}}
"""


def display_request(run_dir, envelope, raw_request):
    """Use a child task's typed scope, never its internal execution prompt."""
    tasks = (envelope or {}).get('tasks') or []
    if Path(run_dir).parent.name == 'child_runs' and len(tasks) == 1 and isinstance(tasks[0], dict):
        label = str(tasks[0].get('label') or '').strip()
        if label:
            return label
    return raw_request


def append_requested_downloads(markdown, run_dir, requested_outputs=None, *, delivery_root=None):
    """Bundle only declared, existing deliverables, preserving their bytes.

    This does not generate a missing result or validate its scientific quality.
    Missing files remain explicit. Copying the evidence into the run makes the
    report usable after downloading the run folder from another computer.
    """
    from .run_layout import resolve_run_path
    root = Path(run_dir).resolve()
    if requested_outputs is None:
        path = resolve_run_path(root, 'request_envelope.json')
        envelope = json.loads(path.read_text()) if path.is_file() else {}
        requested_outputs = envelope.get('final_outputs') or []
    def sha(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                value.update(chunk)
        return value.hexdigest()
    manifest_path = root/'artifacts'/'deliverables'/'manifest.json'
    prior = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest = dict(prior)
    lines = []
    for item in requested_outputs:
        if not isinstance(item, dict) or not item.get('path'):
            continue
        name = str(item['path'])
        path = Path(name).expanduser()
        if delivery_root and path.is_absolute() and path.is_relative_to('/output'):
            path = Path(delivery_root) / path.relative_to('/output')
        elif not path.is_absolute():
            path = root / path
        if path.resolve() in {root/'report.md', root/'report.html'}:
            continue
        label = str(item.get('label') or path.name).replace('\n', ' ').replace('[', '').replace(']', '')
        saved = manifest.get(name) or {}
        bundled = local_report_artifact(root, saved.get('path', '')) if saved else None
        if not path.is_file() and bundled and bundled.is_file() and sha(bundled) == saved.get('sha256'):
            path = bundled
        if path.is_file():
            digest = sha(path)
            destination = root/'artifacts'/'deliverables'/digest/path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(path, destination)
            if sha(destination) != digest:
                raise ValueError('Bundled deliverable differs from its declared source.')
            manifest[name] = {'path': destination.relative_to(root).as_posix(), 'sha256': digest}
            # Evaluation exports already package their evidence in a sibling
            # <filename>.assets directory. Keep that exact relative structure
            # when bringing the declared report into the recipient bundle.
            assets = path.with_name(path.name + '.assets')
            target_assets = destination.with_name(destination.name + '.assets')
            if assets.is_dir():
                if assets.resolve() != target_assets.resolve():
                    shutil.copytree(assets, target_assets, dirs_exist_ok=True)
                manifest[name]['assets'] = {}
                for asset in assets.rglob('*'):
                    if asset.is_file():
                        relative = asset.relative_to(assets)
                        value = sha(asset)
                        if sha(target_assets/relative) != value:
                            raise ValueError('Bundled report evidence differs from its source.')
                        manifest[name]['assets'][relative.as_posix()] = value
            lines.append(f'- [{label}](<{artifact_href(destination, root)}>)')
        else:
            # A previous render may already have bundled this file. Do not
            # search by basename across runs or claim a missing file is present.
            lines.append(f'- {label}: file unavailable in this report bundle.')
    heading = '## Requested output files'
    if not lines:
        return markdown
    if manifest != prior:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    section = heading + '\n\n' + '\n'.join(lines) + '\n'
    if '\n'+heading+'\n' in markdown:
        prefix, _, old = markdown.partition('\n'+heading+'\n')
        _, boundary, suffix = old.partition('\n## ')
        return prefix.rstrip() + '\n\n' + section + ('\n## '+suffix if boundary else '')
    return markdown.rstrip() + '\n\n' + section


def local_report_artifact(root, value):
    """Resolve only existing in-bundle files, including a relocated run tree.

    Rebase an old absolute path only through exact run/child tree components.
    Never search by filename or silently link a different run's evidence.
    """
    root = Path(root).resolve()
    parsed = urlsplit(str(value))
    if parsed.scheme not in {"", "file"} or parsed.netloc:
        return None
    path = Path(unquote(parsed.path))
    candidates = [path if path.is_absolute() else root / path]
    if not path.is_absolute() and len(path.parts) == 1:
        from .run_layout import resolve_run_path
        candidates.append(resolve_run_path(root, path.name))
    if path.is_absolute():
        for index, part in enumerate(path.parts):
            if part == root.name:
                candidates.append(root.joinpath(*path.parts[index + 1:]))
        if "child_runs" in path.parts:
            index = path.parts.index("child_runs")
            candidates.append(root.joinpath(*path.parts[index:]))
    matches = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_relative_to(root) and candidate.exists() and candidate not in matches:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def artifact_href(path, root):
    return quote(Path(os.path.relpath(path, root)).as_posix(), safe="/.")


def read_markdown_fragment(source, report_dir):
    """Rebase a saved fragment's local links when embedding it in a report."""
    source, root = Path(source).resolve(), Path(report_dir).resolve()
    text = source.read_text(encoding="utf-8")
    # Match the same code and link tokens supported by inline_markdown. Code
    # literals, external links and in-document anchors retain their bytes.
    pattern = r"(`+)([^`]*?)\1|\[([^\]]+)\]\((<[^>]+>|[^)]+)\)"
    def rebase(match):
        if match.group(1):
            return match.group(0)
        label, target = match.group(3), match.group(4).strip().strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        candidate = (source.parent / unquote(parsed.path)).resolve()
        if not candidate.is_relative_to(root):
            return label + " (file unavailable)"
        href = artifact_href(candidate, root)
        if parsed.fragment:
            href += "#" + parsed.fragment
        return f"[{label}](<{href}>)"
    return re.sub(pattern, rebase, text)


def _link(label, target, report_dir):
    target = html.unescape(target.strip().strip("<>"))
    parsed = urlsplit(target)
    if parsed.scheme in {"http", "https", "mailto"}:
        href = parsed.geturl()
    elif target.startswith("#"):
        href = target
    elif report_dir and (path := local_report_artifact(report_dir, target)) is not None:
        href = artifact_href(path, report_dir)
    elif parsed.scheme in {"", "file"}:
        # Missing files stay explicit text, never a dead download button.
        return label + ' <span class="unavailable">(file unavailable)</span>'
    else:
        return label
    return '<a href="' + html.escape(href, quote=True) + '">' + label + '</a>'


def inline_markdown(text, report_dir=None):
    # Tokenize code/links before emphasis, so vendor stars and literal markup
    # inside code cannot be reinterpreted as formatting or executable HTML.
    tokens = []
    pattern = r"(`+)([^`]*?)\1|\[([^\]]+)\]\((<[^>]+>|[^)]+)\)|(?<![\w:/])(/[^\s<>`]+)"
    def token(match):
        if match.group(1):
            value = match.group(2)
            result = '<code>' + html.escape(value) + '</code>'
            if report_dir and (path := local_report_artifact(report_dir, value)) is not None and path.is_file():
                if Path(value).is_absolute():
                    result = '<code>' + html.escape(path.name) + '</code>'
                result = _link(result, str(path), report_dir)
        elif match.group(3) is not None:
            result = _link(html.escape(match.group(3)), match.group(4), report_dir)
        else:
            raw = match.group(5)
            value = raw.rstrip('.,;:')
            path = local_report_artifact(report_dir, value) if report_dir else None
            result = (_link(html.escape(path.name), value, report_dir) + html.escape(raw[len(value):])
                      if path and path.is_file() else html.escape(raw))
        tokens.append(result)
        return f"\x00{len(tokens)-1}\x00"
    rendered = html.escape(re.sub(pattern, token, str(text)))
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", rendered)
    for index, value in enumerate(tokens):
        rendered = rendered.replace(f"\x00{index}\x00", value)
    return rendered


def link_child_report_excerpts(markdown, response_payload, run_dir):
    """Replace truncated duplicate excerpts with the full scoped child report.

    The comparison and task outcomes remain in the parent. Full per-task
    scientific details live in their own report, without cached execution text
    being mistaken for the original user request.
    """
    values = list((response_payload.get('completed_task_context') or {}).values())
    values += [task.get('payload') or {} for task in response_payload.get('completed_tasks') or [] if isinstance(task, dict)]
    for value in values:
        if not isinstance(value, dict) or value.get('task_type') != 'design_new_asos':
            continue
        excerpt = value.get('report_excerpt')
        child = local_report_artifact(run_dir, value.get('child_run_dir') or '')
        if not excerpt or excerpt not in markdown or not child or not child.is_dir():
            continue
        path = child/'report.md'
        if not path.is_file():
            continue
        label = str(value.get('task_label') or value.get('label') or 'Full child design report').replace('[', '').replace(']', '').replace('\n', ' ')
        markdown = markdown.replace(excerpt, f'[{label}](<{artifact_href(path, run_dir)}>)')
    return markdown


def table_html(rows, report_dir=None):
    parsed = []
    for row in rows:
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", row.strip()[1:-1])]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        parsed.append(cells)
    if not parsed:
        return ""
    def tr(cells, tag):
        return '<tr>' + ''.join(f'<{tag}>' + inline_markdown(cell.replace('\\|', '|'), report_dir) + f'</{tag}>' for cell in cells) + '</tr>'
    return ('<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable results table"><table><thead>'
            + tr(parsed[0], 'th') + '</thead><tbody>'
            + ''.join(tr(row, 'td') for row in parsed[1:]) + '</tbody></table></div>')


def markdown_html(markdown, report_dir=None):
    lines = str(markdown or '').splitlines()
    blocks, paragraph, bullets = [], [], []
    list_kind = 'ul'
    def flush():
        if paragraph:
            blocks.append('<p>' + ' '.join(paragraph) + '</p>'); paragraph.clear()
        if bullets:
            blocks.append(f'<{list_kind}>' + ''.join('<li>'+s+'</li>' for s in bullets) + f'</{list_kind}>'); bullets.clear()
    index = 0
    while index < len(lines):
        line = lines[index]; stripped = line.strip()
        if not stripped:
            flush(); index += 1; continue
        if stripped.startswith(('```', '~~~')):
            flush(); fence = stripped[:3]; code = []; index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code.append(lines[index]); index += 1
            blocks.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>'); index += 1; continue
        if stripped.startswith('|') and stripped.endswith('|'):
            flush(); rows = []
            while index < len(lines) and lines[index].strip().startswith('|') and lines[index].strip().endswith('|'):
                rows.append(lines[index]); index += 1
            blocks.append(table_html(rows, report_dir)); continue
        heading = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading:
            flush(); n = len(heading[1]); blocks.append(f'<h{n}>' + inline_markdown(heading[2], report_dir) + f'</h{n}>'); index += 1; continue
        bullet = re.match(r'^([-*]|\d+\.)\s+(.+)$', stripped)
        if bullet:
            if paragraph: flush()
            kind = 'ol' if bullet[1][0].isdigit() else 'ul'
            if bullets and kind != list_kind: flush()
            list_kind = kind; bullets.append(inline_markdown(bullet[2], report_dir))
        elif line.startswith('  ') and bullets:
            bullets[-1] += '<br>' + inline_markdown(stripped, report_dir)
        else:
            if bullets: flush()
            paragraph.append(inline_markdown(stripped, report_dir))
        index += 1
    flush()
    return ''.join(blocks)


def final_candidate_panel(run_dir, *, request_envelope=None):
    """Project the delivered inventory, not the broader scored search space."""
    from .run_layout import resolve_run_path
    root = Path(run_dir)
    path = resolve_run_path(root, 'generated_aso_set.json')
    if not path.is_file():
        return ''
    payload = json.loads(path.read_text())
    rows = payload.get('rows')
    if not isinstance(rows, list):
        return ''
    envelope = request_envelope or {}
    if not envelope:
        envelope_path = resolve_run_path(root, 'request_envelope.json')
        if envelope_path.is_file():
            envelope = json.loads(envelope_path.read_text())
    separate_panel = any(isinstance(item, dict) and str(item.get('path') or '').lower().endswith(('.csv', '.tsv'))
                         and not item.get('source_artifact') for item in envelope.get('final_outputs') or [])
    esc = lambda value: html.escape(str(value)) if value is not None and value != '' else 'Not recorded'
    cells = []
    for index, row in enumerate(rows):
        molecule = row.get('molecule_specification') or {}
        annotations = row.get('source_annotations') or {}
        # The target/comparison sequence and generic RC field are not sufficient
        # attestation of the actual physical oligo supplied for synthesis.
        antisense = molecule.get('antisense_5to3') or row.get('synthesized_antisense_sequence')
        rank = row.get('selection_rank')
        identity = row.get('candidate_id') or row.get('name') or annotations.get('Name') or f'Row {index + 1}'
        location = row.get('mature_target_geometry') or row.get('target_geometry') or row.get('genomic_segments')
        if isinstance(location, (dict, list)):
            location = json.dumps(location, ensure_ascii=False)
        if not location and row.get('chrom') and row.get('start') is not None and row.get('end') is not None:
            # Preserve source coordinates; never guess their base convention.
            location = f"{row['chrom']}:{row['start']}–{row['end']} ({row.get('strand') or 'strand not recorded'}; source coordinates)"
        chemistry = molecule.get('architecture') or row.get('chemistry') or annotations.get('chemistry_label') or annotations.get('chemistry')
        status = row.get('recommendation_status') or row.get('mapping_status')
        scope = row.get('job_id') or row.get('program_id') or row.get('route_label')
        rank_text = 'Not recorded' if rank is None else str(rank) + (f' ({scope})' if scope else '')
        values = [identity, rank_text, antisense, chemistry, location, row.get('gc_fraction'), status]
        cells.append('<tr>' + ''.join('<td' + (' class="sequence"' if j == 2 else '') + '>' + esc(value) + '</td>' for j, value in enumerate(values)) + '</tr>')
    headers = ['Candidate', 'Recorded selection rank / scope', 'ASO (5′→3′)', 'Architecture / source chemistry label', 'Target location', 'GC fraction', 'Recorded evidence status']
    links = []
    for name, label in [('generated_aso_set.json', 'Complete selected-panel evidence'), ('design_panel.csv', 'Selected-panel CSV'), ('design_panel.md', 'Selected-panel rationale')]:
        candidate = resolve_run_path(root, name)
        if candidate.is_file():
            links.append('<a href="' + artifact_href(candidate, root) + '">' + label + '</a>')
    title = 'Native design-stage panel' if separate_panel else 'Final selected panel'
    scope_note = ('The separately requested deliverables define the final selection. These native records are supporting evidence and may differ in membership or ordering. '
                  if separate_panel else '')
    return ('<h2>' + title + '</h2><p>' + scope_note + str(len(rows)) + ' delivered native records. Recorded order and ranks are preserved; ranks apply within their recorded scope. These are computational results, not measured activity.</p>'
            + '<div class="table-scroll" tabindex="0" role="region" aria-label="Final selected ASOs"><table><thead><tr>'
            + ''.join('<th>' + h + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(cells) + '</tbody></table></div>'
            + '<div class="downloads">' + ''.join(links) + '</div>')
