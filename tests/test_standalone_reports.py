"""Reading copies work independently and never collect reference-sized assets."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil

from aso_agents.requested_artifacts import copy_native_export
from aso_agents.report_exports import export_native_report
from aso_agents.report_presentation import append_requested_downloads
from aso_agents import standalone_report


def packet(path):
    return json.loads(re.search(r'<script id="modal-portable-assets" type="application/json">(.*?)</script>', path.read_text(), re.S)[1])


def test_single_html_keeps_exact_downloads_and_scientific_text_after_relocation(tmp_path):
    run = tmp_path/'run'; run.mkdir()
    data = b'id,sequence,value\r\na,ACGT,0\r\nb,GGTT,\r\n'
    (run/'panel.csv').write_bytes(data)
    literal = '<script type="application/json" id="report-data">{"score":0,"unknown":null}</script>'
    source = run/'report.html'
    source.write_text('<html><body><h1>Recorded results: 0</h1>'+literal+'<a href="panel.csv">ASOs</a></body></html>')
    before = source.read_bytes()
    dest = tmp_path/'delivery'/'report.html'
    receipt = copy_native_export(source, dest)
    assert receipt['method'] == 'standalone_html'
    assert literal in dest.read_text() and '<h1>Recorded results: 0</h1>' in dest.read_text()
    assert source.read_bytes() == before
    assert list(dest.parent.iterdir()) == [dest]
    moved = tmp_path/'moved'; shutil.move(dest.parent, moved); shutil.rmtree(run)
    files = list(packet(moved/dest.name)['files'].values())
    assert len(files) == 1 and base64.b64decode(files[0]['base64']) == data
    assert files[0]['sha256'] == hashlib.sha256(data).hexdigest()


def test_reference_assets_and_large_evidence_are_excluded_before_read_or_hash(tmp_path, monkeypatch):
    run = tmp_path/'run'; run.mkdir()
    blocked = [run/name for name in ['human.fasta.gz', 'human.1.bt2', 'weights.pt', 'evidence.json']]
    for path in blocked:
        with path.open('wb') as handle:
            handle.truncate(4*1024*1024*1024)
    original_read = Path.read_bytes
    original_open = Path.open
    def checked_open(path, *args, **kwargs):
        assert path not in blocked, f'Reference or large evidence was opened: {path}'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', checked_open)
    source = run/'report.html'; source.write_text('<html><body>'+''.join(f'<a href="{p.name}">{p.name}</a>' for p in blocked)+'</body></html>')
    dest = tmp_path/'delivery'/'report.html'
    receipt = copy_native_export(source, dest)
    assert receipt['embedded_bytes'] == 0 and len(receipt['omitted_links']) == 4
    assert dest.stat().st_size < 10000
    md = run/'report.md'; md.write_text('\n'.join(f'[Evidence]({p.name})' for p in blocked))
    bounded = export_native_report(md, tmp_path/'delivery'/'report.md', source_root=run)
    assert bounded['asset_count'] == 1 and len(bounded['unresolved_links']) == 4


def test_many_small_downloads_cannot_exceed_the_total_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(standalone_report, 'MAX_DOWNLOAD_TOTAL_BYTES', 16)
    for name in ['a.csv', 'b.csv', 'c.csv']:
        (tmp_path/name).write_bytes(name[0].encode() * 8)
    source = tmp_path/'report.html'; source.write_text('<html><body>'+''.join(f'<a href="{n}.csv">{n}</a>' for n in 'abc')+'</body></html>')
    _, receipt = standalone_report.standalone_html(source.read_text(), source_path=source, source_root=tmp_path)
    assert receipt['embedded_bytes'] == 16
    assert receipt['omitted_links'] == [{'name':'c.csv','reason':'download_size_limit','bytes':8}]


def test_identical_downloads_share_content_and_preserve_all_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(standalone_report, 'MAX_DOWNLOAD_TOTAL_BYTES', 8)
    for name in ('a.csv','b.csv'):
        (tmp_path/name).write_bytes(b'12345678')
    source=tmp_path/'report.html'
    document='<html><body><a href="a.csv">First</a><a href="b.csv">Second</a></body></html>'
    page,receipt=standalone_report.standalone_html(document,source_path=source,source_root=tmp_path)
    payload=json.loads(re.search(r'<script id="modal-portable-assets" type="application/json">(.*?)</script>',page,re.S)[1])
    assert receipt['embedded_bytes']==8 and receipt['omitted_links']==[]
    assert len(payload['files'])==1
    assert {'a.csv','b.csv'}<=set(next(iter(payload['files'].values()))['aliases'])


def test_primary_downloads_get_the_budget_before_earlier_supporting_links(tmp_path, monkeypatch):
    monkeypatch.setattr(standalone_report, 'MAX_DOWNLOAD_TOTAL_BYTES', 16)
    for name in ['detail_a.csv', 'detail_b.csv', 'panel.csv']:
        (tmp_path/name).write_bytes(name.encode()[:8].ljust(8, b'0'))
    source = tmp_path/'report.html'
    source.write_text('<html><body><p>Scientific result: 0</p><a href="detail_a.csv">First details</a>'
                      '<a href="detail_b.csv">Second details</a>'
                      '<a data-download-priority="primary" href="panel.csv">Selected ASOs</a></body></html>')
    page, receipt = standalone_report.standalone_html(source.read_text(), source_path=source, source_root=tmp_path)
    assert receipt['embedded_bytes'] == 16
    assert receipt['omitted_links'] == [{'name': 'detail_b.csv', 'reason': 'download_size_limit', 'bytes': 8}]
    packet = json.loads(re.search(r'<script id="modal-portable-assets" type="application/json">(.*?)</script>', page, re.S)[1])
    assert {item['name'] for item in packet['files'].values()} == {'panel.csv', 'detail_a.csv'}
    assert 'Scientific result: 0' in page
    assert page.index('First details') < page.index('Second details') < page.index('Selected ASOs')
    assert re.search(r'href="#portable-file-[^"]+">Selected ASOs</a>', page)


def test_legacy_bundle_does_not_copy_unlinked_genome_or_symlink(tmp_path):
    source = tmp_path/'legacy'/'report.md'; source.parent.mkdir()
    source.write_text('[Data](report.md.assets/data.csv)')
    assets = source.with_name(source.name+'.assets'); assets.mkdir()
    (assets/'data.csv').write_bytes(b'unchanged,0\n')
    (assets/'genome.fa').write_text('large resource')
    (assets/'alias').symlink_to(tmp_path, target_is_directory=True)
    receiver = tmp_path/'receiver'; receiver.mkdir()
    append_requested_downloads('# Results', receiver, [{'path':str(source)}])
    assert not list(receiver.rglob('*.fa'))
    assert not any(p.is_symlink() for p in receiver.rglob('*'))
    assert any(p.read_bytes()==b'unchanged,0\n' for p in receiver.rglob('*.csv'))


def test_final_delivery_refreshes_stale_warnings_and_receipt_hashes(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    from aso_agents.html_report import render_goal_intake_html
    run = tmp_path/'run'; run.mkdir()
    md = run/'report.md'; md.write_text('# Results\nScore: 0\n\n## Requested output files\n- ASOs: file unavailable in this report bundle.\n')
    html = run/'report.html'; html.write_text(render_goal_intake_html({'meta':{'gene_symbol':'GENE'}, 'overview':{'report_markdown':md.read_text()}, 'oligoai_report_dir':str(run)}))
    csv = run/'panel.csv'; csv.write_bytes(b'id,value\r\na,0\r\n')
    catalog = {'report_html':str(html), 'report_md':str(md), 'design_panel_csv':str(csv)}
    monkeypatch.setattr(pipeline, '_native_export_catalog', lambda *args:catalog)
    monkeypatch.setattr(pipeline, '_run_sandbox_analysis_sync', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('No model needed')))
    outputs = [{'path':str(run/f'export.{extension}'), 'kind':'file', 'source_artifact':key} for extension,key in [('html','report_html'),('md','report_md'),('csv','design_panel_csv')]]
    receipt = pipeline._materialize_requested_final_outputs({'request_envelope':{'final_outputs':outputs}},str(run))
    assert receipt['ok'] and not receipt['execution_attempts']
    for path in [md, html, run/'export.html', run/'export.md']:
        assert 'file unavailable in this report bundle' not in path.read_text()
        assert 'Score: 0' in path.read_text()
    assert (run/'export.csv').read_bytes() == csv.read_bytes()
    assert receipt['output_sha256'] == {str(run/f'export.{ext}'):hashlib.sha256((run/f'export.{ext}').read_bytes()).hexdigest() for ext in ['html','md','csv']}
    assert any(base64.b64decode(item['base64']) == csv.read_bytes() for item in packet(run/'export.html')['files'].values())


def test_real_missing_files_remain_visible_when_refreshing_downloads(tmp_path):
    result = append_requested_downloads('# Results', tmp_path, [{'path':'missing.csv','label':'ASOs'}])
    assert 'ASOs: file unavailable' in result


def test_omitted_large_iframe_cannot_load_the_parent_report_recursively(tmp_path):
    child = tmp_path/'large.html'
    with child.open('wb') as handle:
        handle.truncate(3*1024*1024)
    source = tmp_path/'report.html'
    source.write_text('<html><body><iframe src="large.html"></iframe><a href="large.html">Child</a></body></html>')
    destination = tmp_path/'export'/'report.html'
    copy_native_export(source, destination)
    result = destination.read_text()
    assert '<iframe src="data:"></iframe>' not in result
    assert '<iframe src="data:,"></iframe>' in result
    assert '<a href="#files-kept-in-run">Child</a>' in result
