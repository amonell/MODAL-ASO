"""Frozen MAPT examples: real saved geometry/matches, no network or model calls."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

import run_goal_intake as runtime
from aso_agents.exon_identity_clarification import require_resolved_exon_identities
from aso_agents.gapmer_target_groups import resolve_groups
from aso_agents.report_explorer_data import project_records
from aso_agents.report_exon_names import project_exon_names
from aso_agents.report_selection_exports import SELECTION_EXPORT_SCRIPT


def regression(tmp_path):
    fixture=json.loads((Path(__file__).parent/'fixtures/mapt_group_regression.json').read_text())
    goal={'gene_symbol':'MAPT','species':'human','intent':'knockdown',
        'transcript_lock':{'kind':'explicit_id','value':'ENST00000351559.10'},
        'isoform_transcripts':{'reference':fixture['reference']},
        'splice_isoform_objectives':[{'objective_scope':'aggregate_class','isoform_label':'exon 10 included',
            'reference_transcript_id':'ENST00000351559.10','desired_abundance':'decrease','role':'context',
            'aggregate_class':{'basis':'exon_presence','exon_label':10,'exon_index':9,'presence':'included'}}]}
    require_resolved_exon_identities(goal)
    groups=resolve_groups(goal,fixture['catalog'],goal['isoform_transcripts'])
    source=tmp_path/'tiles.json';source.write_text(json.dumps({'rows':fixture['rows']}))
    matches={len(r['Sequence']):{} for r in fixture['rows']}
    for row in fixture['rows']:matches[len(row['Sequence'])][row['Sequence']]=set(row['isoform_uniqueness_labels'])
    scored=runtime._annotate_gapmer_uniqueness(str(source),matches,None,len(fixture['catalog']),
        transcript_catalog=fixture['catalog'],resolved_transcripts=goal['isoform_transcripts'],structured_goal=goal)
    rows=json.loads(Path(scored).read_text())['rows']
    return fixture,goal,groups,rows


def test_saved_mapt_three_examples_through_group_scoring_and_report_export(tmp_path):
    fixture,goal,groups,rows=regression(tmp_path)
    group=groups['groups'][0]
    assert [len(group[k]) for k in ('members','nonmembers','unknown')]==[13,7,10]
    by_name={r['Name']:r for r in rows}
    expectations=[('4R tau_cdna1021-1040',13,0,2,13/15,1),
                  ('4R tau_cdna1100-1119',13,6,2,13/21,1),
                  ('4R tau_cdna3358-3377',3,0,0,1,3/13)]
    for name,included,skipped,unknown,selectivity,coverage in expectations:
        row=by_name[name]
        assert [row[k] for k in ('target_isoform_match_count','other_isoform_match_count','unclassified_isoform_match_count')]==[included,skipped,unknown]
        assert row['isoform_uniqueness_score']==pytest.approx(selectivity)
        assert row['target_group_coverage_fraction']==pytest.approx(coverage)
    named=project_exon_names(goal['isoform_transcripts']['reference'])
    assert named['exons'][8]['label']=='Requested exon 10'
    assert named['exons'][8]['verified_exon_label'] is None
    assert named['exons'][9]['label']=='tenth annotated exon'
    packet=project_records(rows)
    selectivity=next(a for a in packet['attributes'] if a['id']=='isoform_selectivity')
    coverage=next(a for a in packet['attributes'] if a['id']=='target_group_coverage')
    assert selectivity['direction']=='higher' and coverage['direction']=='higher'
    assert sorted(selectivity['values'])==pytest.approx(sorted([13/15,13/21,1]))
    script=SELECTION_EXPORT_SCRIPT+'\nconst records='+json.dumps(packet['records'])+';\n'+'''
const byMolecule=new Map(records.map(r=>[r.molecule,[r]]));
console.log(JSON.stringify(selectedMoleculeCsv(records,byMolecule,new Map(),{exportAttributes:()=>[]})));
'''
    export=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert len(export)==4
    assert all(json.loads(row[-1])[0]['aso'] for row in export[1:])
    # Source evidence remains byte-equivalent after scoring; old scores are provenance.
    assert json.loads((tmp_path/'tiles.json').read_text())['rows']==fixture['rows']


def test_saved_mapt_group_metric_filters_and_colors_in_browser(tmp_path):
    import shutil
    playwright=pytest.importorskip('playwright.sync_api')
    executable=shutil.which('google-chrome') or shutil.which('chromium')
    if not executable:pytest.skip('Local browser unavailable')
    from aso_agents.readable_report import basic_document
    _,_,_,rows=regression(tmp_path)
    packet=project_records(rows)
    document=basic_document('MAPT target-group regression','# Results\nSaved sequence matches.',candidate_data=packet)
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox'])
        page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_content(document)
        page.locator('[data-attribute-action="quick-color"][data-attribute-id="isoform_selectivity"]').click()
        assert page.locator('[data-role=attribute-color-label]').inner_text()=='Target selectivity'
        colors=page.locator('.report-attribute-swatch').evaluate_all('(els)=>els.map(el=>getComputedStyle(el).backgroundColor)')
        assert len(set(colors))==3
        for attribute,minimum in [('isoform_selectivity','.8'),('target_group_coverage','.9')]:
            page.get_by_role('button',name='Add filter',exact=True).click()
            page.locator('[data-role=attribute-options]').select_option(attribute)
            page.locator('[data-attribute-action=choose]').click()
            page.locator(f'[data-filter-id="{attribute}"] [data-filter-input=low]').fill(minimum)
        assert page.locator('[data-role=attribute-summary]').inner_text().startswith('1 of 3 ASOs match.')
        assert page.locator('tbody tr[data-record]').count()==1
        assert page.locator('tbody tr[data-record]').get_attribute('data-record')==next(
            row['id'] for row in packet['records'] if row['name']=='4R tau_cdna1021-1040')
        assert not errors
        browser.close()
