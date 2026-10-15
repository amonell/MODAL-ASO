"""Recipient handoff checks for exported reports, independent of any target."""
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

import pytest

from aso_agents.requested_artifacts import copy_native_export
from aso_agents.report_exports import export_native_report
from aso_agents.report_presentation import append_requested_downloads


def put(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


def linked(report, label):
    match = re.search(r"\[" + re.escape(label) + r"\]\(<?([^)>]+)>?\)", report.read_text())
    assert match, report.read_text()
    return (report.parent / unquote(urlsplit(match[1]).path)).resolve()


def test_named_markdown_export_survives_moving_without_the_source_run(tmp_path):
    run = tmp_path / "producer"
    observations = b"id,measured,unknown\r\na,0,\r\n"
    evidence = put(run, "tables/data & evidence.csv", observations)
    text = '# Recorded results\r\n\r\nMeasured value: 0. Unknown remains unknown.\r\n[Data](<tables/data & evidence.csv#row-a>)\r\n'
    source = put(run, "report.md", text)
    dest = tmp_path / "exports" / "named report.md"
    before = source.read_bytes()
    receipt = copy_native_export(source, dest, source_root=run)
    assert source.read_bytes() == before and evidence.read_bytes() == observations
    assert receipt['method'] == 'native_report_bundle'
    assert receipt['source_sha256'] == hashlib.sha256(before).hexdigest()
    assert receipt['sha256'] == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert b'Measured value: 0. Unknown remains unknown.\r\n' in dest.read_bytes()
    assert '#row-a>' in dest.read_text()
    assert receipt['unresolved_links'] == []
    relocated = tmp_path / "recipient"
    shutil.move(str(dest.parent), relocated)
    shutil.rmtree(run)
    assert linked(relocated / dest.name, 'Data').read_bytes() == observations


def test_nested_reports_cycles_and_repeated_basenames_keep_their_own_evidence(tmp_path):
    run = tmp_path / 'run'
    source = put(run, 'report.md', '[Left](left/report.md) [Right](right/report.md)')
    put(run, 'left/report.md', '[Data](evidence.json) [Back](../report.md#results)')
    put(run, 'right/report.md', '[Data](evidence.json) [Left](../left/report.md)')
    put(run, 'left/evidence.json', '{"value": 0}')
    put(run, 'right/evidence.json', '{"value": null}')
    dest = tmp_path / 'delivery' / 'summary.md'
    receipt = copy_native_export(source, dest, source_root=run)
    assert receipt['asset_count'] == 5
    left, right = linked(dest, 'Left'), linked(dest, 'Right')
    assert json.loads(linked(left, 'Data').read_text()) == {'value': 0}
    assert json.loads(linked(right, 'Data').read_text()) == {'value': None}
    assert linked(right, 'Left') == left
    assert linked(linked(left, 'Back'), 'Left') == left


def test_html_attributes_are_rebased_without_rewriting_script_or_visible_text(tmp_path):
    run = tmp_path / 'run'
    data = put(run, 'data & evidence.json', '{}')
    put(run, 'image.png', b'\x89PNG\x00\xff')
    put(run, 'nested/view.html', '<a href="../report.html#main">Back</a>')
    literal = '<script>const example = "<a href=\'private.csv\'>";</script>'
    text = ('<!doctype html><h1 id="main">Recorded: 0</h1>' + literal
            + f'<a title="href=\'private.csv\'" href="{data}?a=1&amp;b=2#row">Evidence</a>'
            + '<img src=image.png><iframe src="nested/view.html"></iframe>')
    source = put(run, 'report.html', text)
    dest = tmp_path / 'out' / 'result.html'
    receipt = copy_native_export(source, dest, source_root=run)
    result = dest.read_text()
    assert literal in result and '<h1 id="main">Recorded: 0</h1>' in result
    assert 'title="href=\'private.csv\'"' in result
    assert '?a=1&amp;b=2#row' in result
    assert str(run) not in result
    files = Path(receipt['manifest']).parent / 'files'
    assert (files / data.name).read_bytes() == b'{}'
    assert (files / 'image.png').read_bytes() == b'\x89PNG\x00\xff'
    assert '<a href="../report.html#main">Back</a>' == (files / 'nested/view.html').read_text()
    assert source.read_text() == text


def test_markdown_code_titles_reference_definitions_and_external_links(tmp_path):
    run = tmp_path / 'run'
    put(run, 'data (one).csv', 'a,0\n')
    protected = ('`[literal](missing.csv)`\n``[literal](missing.csv)``\n'
                 '```text\n[data](missing.csv)\n```\n'
                 '~~~text\n[data](missing.csv)\n~~~\n'
                 '    [data](missing.csv)\n'
                 '[External](https://example.org/data) [Anchor](#section)\n')
    source = put(run, 'report.md', protected + '[Data](data%20%28one%29.csv "Measured data")\n'
                 '[Ref][observations]\n[observations]: <data (one).csv> "Full evidence"\n')
    dest = tmp_path / 'out' / 'report.md'
    receipt = copy_native_export(source, dest, source_root=run)
    assert dest.read_text().startswith(protected)
    assert '"Measured data"' in dest.read_text() and '"Full evidence"' in dest.read_text()
    assert receipt['asset_count'] == 2 and receipt['unresolved_links'] == []
    assert (Path(receipt['manifest']).parent/'files'/'data (one).csv').read_text() == 'a,0\n'


def test_missing_outside_and_symlink_references_are_disclosed_not_replaced_by_name(tmp_path):
    run = tmp_path / 'run'
    private = put(tmp_path, 'private.csv', 'not part of this run')
    source = put(run, 'report.md', '[Missing](missing.csv) [Outside](../private.csv) [Alias](alias.csv)')
    (run/'alias.csv').symlink_to(private)
    put(tmp_path/'other-run', 'missing.csv', 'different evidence')
    dest = tmp_path/'out'/'report.md'
    receipt = copy_native_export(source, dest, source_root=run)
    assert receipt['asset_count'] == 1
    assert [r['reason'] for r in receipt['unresolved_links']] == [
        'file_unavailable', 'outside_producing_run', 'outside_producing_run']
    assert not linked(dest, 'Missing').exists()
    assert not list(dest.with_name(dest.name+'.assets').rglob('*.csv'))
    assert private.read_text() == 'not part of this run'


def test_export_is_repeatable_and_new_evidence_retains_the_previous_bundle(tmp_path):
    run = tmp_path/'run'
    source = put(run, 'report.md', '[Data](evidence.csv)')
    data = put(run, 'evidence.csv', 'old evidence')
    dest = tmp_path/'out'/'report.md'
    first = copy_native_export(source, dest)
    assert copy_native_export(source, dest) == first
    old = linked(dest, 'Data')
    data.write_text('new evidence')
    second = copy_native_export(source, dest)
    assert second['manifest'] != first['manifest']
    assert old.read_text() == 'old evidence'
    assert linked(dest, 'Data').read_text() == 'new evidence'


def test_tampered_content_addressed_asset_is_not_silently_overwritten(tmp_path):
    run = tmp_path/'run'
    source = put(run, 'report.md', '[Data](evidence.csv)')
    put(run, 'evidence.csv', 'recorded evidence')
    dest = tmp_path/'out'/'report.md'
    copy_native_export(source, dest)
    asset = linked(dest, 'Data')
    asset.write_text('altered evidence')
    with pytest.raises(ValueError, match='native_report_export_integrity_error'):
        copy_native_export(source, dest)
    assert asset.read_text() == 'altered evidence'
    assert source.read_text() == '[Data](evidence.csv)'


def test_requested_download_bundle_keeps_the_export_and_its_assets_portable(tmp_path):
    run = tmp_path/'run'
    source = put(run, 'report.md', '[Data](evidence.json)')
    put(run, 'evidence.json', '{"result": 0}')
    dest = tmp_path/'out'/'named.md'
    copy_native_export(source, dest)
    receiver = tmp_path/'recipient'; receiver.mkdir()
    append_requested_downloads('# Results', receiver, [{'path': str(dest)}])
    bundled = next((receiver/'artifacts/deliverables').rglob('named.md'))
    assert bundled.read_bytes() == dest.read_bytes()
    shutil.rmtree(run); shutil.rmtree(dest.parent)
    assert json.loads(linked(bundled, 'Data').read_text()) == {'result': 0}


@pytest.mark.parametrize('name,content', [('panel.csv', b'id,value\r\na,0\r\n'),
                                        ('records.json', b'{ "unknown": null, "measured": 0 }')])
def test_non_report_exports_keep_exact_native_bytes(tmp_path, name, content):
    source = put(tmp_path/'run', name, content)
    dest = tmp_path/'out'/name
    receipt = copy_native_export(source, dest)
    assert dest.read_bytes() == source.read_bytes() == content
    assert receipt['method'] == 'native_artifact_copy'
    assert not dest.with_name(dest.name+'.assets').exists()


def test_in_place_copy_does_not_modify_the_canonical_report(tmp_path):
    source = put(tmp_path, 'report.md', '[Data](missing.csv)')
    before = source.read_bytes()
    assert copy_native_export(source, source)['method'] == 'native_artifact_copy'
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match='would_replace_source'):
        export_native_report(source, source, source_root=tmp_path)


def test_relative_caller_paths_keep_the_delivery_receipt_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = put(Path('run'), 'report.md', '[Data](data.csv)')
    put(Path('run'), 'data.csv', 'measured,0\n')
    destination = Path('delivery/named.md')
    receipt = copy_native_export(source, destination)
    assert receipt['source'] == str(source)
    assert receipt['destination'] == str(destination)
    assert linked(destination, 'Data').read_text() == 'measured,0\n'


def test_native_delivery_passes_the_producing_run_scope_for_child_reports(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    run = tmp_path/'run'
    child = put(run, 'child_runs/arm/report.md', '[Shared](../../shared.json)')
    put(run, 'shared.json', '{"count": 0}')
    dest = tmp_path/'delivery'/'report.md'
    monkeypatch.setattr(pipeline, '_native_export_catalog', lambda *args: {'report_md': str(child)})
    def no_model(*args, **kwargs):
        pytest.fail('An exact native report export must not call a model.')
    monkeypatch.setattr(pipeline, '_run_sandbox_analysis_sync', no_model)
    state = {'request_envelope': {'tasks': [{'task_id': 'arm', 'task_type': 'analyze_artifact'}],
             'final_outputs': [{'task_id': 'arm', 'kind': 'file', 'source_artifact': 'report_md', 'path': str(dest)}]}}
    receipt = pipeline._materialize_requested_final_outputs(state, str(run))
    assert receipt['ok'] and receipt['execution_attempts'] == []
    assert json.loads(linked(dest, 'Shared').read_text()) == {'count': 0}
