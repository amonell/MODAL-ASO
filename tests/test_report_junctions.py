from copy import deepcopy
import json
import subprocess

import pytest

from aso_agents.report_explorer_data import project_run, safeguard_summary, site_name
from aso_agents.report_junctions import junction_measurements, junction_attributes
from aso_agents.report_selection_exports import SELECTION_EXPORT_SCRIPT
from test_report_explorer import run_fixture, save


def point(kind='acceptor', position=110, reference='TX.2', values=(-.2, -.4), **extra):
    return {'canonical_label':kind+'@'+str(position),'site_kind':kind,
        kind:{'chrom':'chr2','position':position,'strand':'-'},
        'reference_transcript_id':reference,'reference_binding_status':'resolved_catalog_identity',
        'evaluation_status':'evaluated',f'delta_{kind}_deletion':values[0],
        f'delta_{kind}_replacement':values[1],**extra}


def test_junctions_keep_identity_independent_of_response_view():
    entries=[point(),point('donor',90),point(position=210,reference='TX.3'),point('donor',190)]
    row={'splice_site_deltas':entries}
    before=deepcopy(row)
    a=junction_measurements(row,{'id':'acceptor','job':'j','transcript':'TX.2'}, {}, site_name)
    b=junction_measurements(row,{'id':'donor','job':'j','transcript':'TX.2'}, {}, site_name)
    assert a==b and len(a)==4 and row==before
    attrs=junction_attributes([{'junction_measurements':a},{'junction_measurements':b}])
    assert len(attrs)==4 and all(attr['values'][0]==attr['values'][1] for attr in attrs)
    other=junction_measurements(row,{'id':'other','job':'another-job'}, {}, site_name)
    assert {p['id'] for p in a}.isdisjoint(p['id'] for p in other)


@pytest.mark.parametrize('values,status', [((.1,None),'evaluated'),((None,None),'evaluated'),((.1,.2),'failed')])
def test_missing_or_failed_measurements_are_unavailable(values,status):
    result=junction_measurements({'splice_site_deltas':[point(values=values,evaluation_status=status)]},
        {'id':'a'}, {}, site_name)
    assert result[0]['mean'] is None and result[0]['status']=='unavailable'


def test_duplicate_conflicts_and_reference_versions_are_preserved():
    records=[point(),point(),point(values=(.4,.5)),point(reference='TX.3')]
    result=junction_measurements({'splice_site_deltas':records},{'id':'a'}, {},site_name)
    assert len(result)==2
    assert result[0]['status']=='conflicting_records' and result[0]['mean'] is None
    assert len(result[0]['variants'])==2
    attrs=junction_attributes([{'junction_measurements':result}])
    assert attrs[0]['label']!=attrs[1]['label']


def test_main_signed_change_matches_named_junction_for_decrease_goal(tmp_path):
    goal,raw,pool=run_fixture(tmp_path)
    goal['design_programs'][0]['desired_delta']='decrease'
    for row in raw:
        row['splice_site_deltas'][0].update(delta_acceptor_deletion=-.2,delta_acceptor_replacement=-.4)
    pool[0].update(raw[0])
    save(tmp_path,'artifacts/intake/structured_goal.json',goal)
    save(tmp_path,'artifacts/design/steric_jobs/j/scored_tiles.json',{'rows':raw})
    save(tmp_path,'artifacts/candidates/candidate_assessments.json',{'rows':pool})
    data=project_run(tmp_path)
    assert all(row['effect'] < 0 for row in data['records'])
    attr=next(a for a in data['attributes'] if a['group']=='Splice junction predictions')
    assert attr['values']==[r['effect'] for r in data['records']]


@pytest.mark.parametrize('required', [False,True,None])
def test_safeguards_keep_direction_enforcement_and_individual_failures(required):
    guard={'label':'donor@150','status':'fail','passes':False,'required':required,
        'max_increase':.05,'delta_deletion':.4883309304714203,'delta_replacement':-.42}
    before=deepcopy(guard)
    view=safeguard_summary({'safeguard_report':[guard]}, {'donor@150':{'site_kind':'donor','position':150}})
    detail=view['details'][0]
    assert guard==before and view['status']=='review'
    assert detail['required'] is required and detail['bounds']['max_increase']==.05
    assert detail['simulations'][0]['value']==.4883309304714203
    assert detail['simulations'][1]['value']==-.42
    assert ('Review-only check' if required is False else 'Required check' if required else 'Enforcement unrecorded') in detail['text']
    # An average inside the bound leaves the recorded per-simulation failure intact.
    assert sum(s['value'] for s in detail['simulations'])/2 < .05


def test_selection_csv_groups_molecules_preserves_all_contexts_and_precision(tmp_path):
    from aso_agents.report_explorer_data import project_records
    data=project_records([{'antisense_5to3':'ACGT','ordering_notation':'A*C*G*T'},
                          {'antisense_5to3':'ACGT','ordering_notation':'ACGT'}])
    first,other=data['records']
    first.update(effect=-.12345678912345678,context='acceptor',native_rank=3,gc=.5)
    second={**deepcopy(first),'id':'second-context','context':'donor','effect':.4883309304714203,'native_rank':8,'gc':.5000000001}
    other['context']='donor'
    rows=[first,second,other]
    script=SELECTION_EXPORT_SCRIPT+'\nconst rows='+json.dumps(rows)+';\n'+'''
const byMolecule=new Map();rows.forEach(r=>{if(!byMolecule.has(r.molecule))byMolecule.set(r.molecule,[]);byMolecule.get(r.molecule).push(r);});
const contexts=new Map([['acceptor',{id:'acceptor',label:'Acceptor'}],['donor',{id:'donor',label:'Donor'}]]);
const attributes={exportAttributes:()=>[],allAttributes:()=>[{id:'hidden-score',label:'Saved score',unit:''}],values:()=>[.9876543211234567],exportValue:()=>''};
console.log(JSON.stringify({csv:selectedMoleculeCsv([rows[2],rows[0]],byMolecule,contexts,attributes),
evidence:selectedMoleculeEvidence([rows[2],rows[0]],byMolecule,contexts,attributes)}));
'''
    path=tmp_path/'export.js';path.write_text(script)
    result=json.loads(subprocess.check_output(['node',str(path)],text=True))
    headers,*records=result['csv']
    assert len(records)==2 and all(len(r)==len(headers) for r in records)
    exported=[dict(zip(headers,r)) for r in records]
    assert exported[0]['ordering_notation']=='ACGT' and exported[1]['ordering_notation']=='A*C*G*T'
    assert exported[1]['native_rank']=='' and exported[1]['predicted_splice_effect']==''
    evidence=json.loads(exported[1]['context_evidence_json'])
    assert [r['native_rank'] for r in evidence]==[3,8]
    assert [r['effect'] for r in evidence]==[first['effect'],second['effect']]
    assert [r['gc'] for r in evidence]==[.5,.5000000001]
    assert result['evidence']['attributes'][0]['id']=='hidden-score'
    assert result['evidence']['attributes'][0]['values']==[[.9876543211234567]]*3
    assert exported[1]['full_context_evidence_file']=='ASO_selection_contexts.json'


def test_independent_junction_filters_context_switch_and_single_row_download(tmp_path):
    import csv
    import io
    import shutil
    playwright=pytest.importorskip('playwright.sync_api')
    executable=shutil.which('google-chrome') or shutil.which('chromium')
    if not executable:pytest.skip('Local browser unavailable')
    from aso_agents.readable_report import basic_document
    goal,raw,pool=run_fixture(tmp_path)
    goal['design_programs'][0]['desired_delta']='decrease'
    for row,values in zip(raw,[(-.2,.04,.02,-.03),(-.3,.1,.02,-.03),(-.11,-.05,.07,-.03)]):
        row['splice_site_deltas']=[point(kind,pos,values=(value,value)) for (kind,pos),value in zip(
            [('acceptor',110),('donor',150),('acceptor',210),('donor',250)],values)]
    pool=[{**row,'job_id':'j','program_id':'p','route_label':route} for row in raw for route in ('acceptor@110','donor@150')]
    save(tmp_path,'artifacts/intake/structured_goal.json',goal)
    save(tmp_path,'artifacts/design/steric_jobs/j/scored_tiles.json',{'rows':raw})
    save(tmp_path,'artifacts/candidates/candidate_assessments.json',{'rows':pool})
    data=project_run(tmp_path)
    attributes={a['reference_scope']['position']:a for a in data['attributes'] if a.get('group')=='Splice junction predictions'}
    assert len(attributes)==4
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox'])
        page=browser.new_page(accept_downloads=True);errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_content(basic_document('Four junctions','# Results\nRecorded splice measurements.',candidate_data=data))
        assert page.get_by_label('Maximum predicted splice effect',exact=True).is_visible()
        assert page.locator('tbody tr[data-record]').first.locator('[data-column=effect]').inner_text()=='-0.300'
        for position,low,high in [(110,-1,-.1),(150,-1,.05),(210,-.05,.05),(250,-.05,.05)]:
            page.get_by_role('button',name='Junction filters',exact=True).click()
            page.locator('[data-role=attribute-options]').select_option(attributes[position]['id'])
            page.locator('[data-attribute-action=choose]').click()
            card=page.locator('[data-filter-id="'+attributes[position]['id']+'"]')
            card.locator('[data-filter-input=low]').fill(str(low))
            card.locator('[data-filter-input=high]').fill(str(high))
        assert page.locator('[data-role=attribute-summary]').inner_text().startswith('1 of 3 ASOs match.')
        page.locator('[data-pick]').check()
        other=next(c['id'] for c in data['contexts'] if c['site']=='donor@150')
        page.locator('[data-role=context]').select_option(other)
        assert page.locator('[data-role=attribute-summary]').inner_text().startswith('1 of 3 ASOs match.')
        assert page.locator('tbody tr[data-record]').count()==1
        assert page.locator('[data-role=basket-count]').inner_text()=='1 ASO selected'
        with page.expect_download() as downloading:
            page.locator('[data-action=csv]').click()
        target=tmp_path/'basket.csv';downloading.value.save_as(target)
        exported=list(csv.DictReader(io.StringIO(target.read_text())))
        assert len(exported)==1 and exported[0]['context_count']=='2'
        evidence=json.loads(exported[0]['context_evidence_json'])
        assert sorted(r['effect'] for r in evidence)==[-.2,.04]
        assert all(len(r['junction_measurements'])==4 for r in evidence)
        page.locator('[data-filter-id="'+attributes[110]['id']+'"] [data-filter-input=high]').fill('-.25')
        assert '0 of 3 ASOs match.' in page.locator('[data-role=attribute-summary]').inner_text()
        page.locator('[data-action=empty-reset]').click()
        assert page.locator('tbody tr[data-record]').count()==3
        for width in (390,1440):
            page.set_viewport_size({'width':width,'height':900})
            assert page.evaluate('document.documentElement.scrollWidth')<=width
        assert not errors
        browser.close()
