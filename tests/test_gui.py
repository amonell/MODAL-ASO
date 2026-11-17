"""The browser host uses the same CLI and preserves explicit human answers."""
import json
from pathlib import Path
import threading
import time
import subprocess
import urllib.error
import urllib.request

import pytest

from aso_agents.gui import Jobs, make_server
from aso_agents.gui_worker import Host, read_json, run, write_json

REAL_POPEN = subprocess.Popen


def worker_process(command, *args, **kwargs):
    return Process(command, *args, **kwargs) if 'aso_agents.gui_worker' in command else REAL_POPEN(command, *args, **kwargs)


class Process:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.returncode, self.pid = args, kwargs, None, 123456789
    def poll(self):
        return self.returncode
    def wait(self, **kwargs):
        self.returncode = 0
        return 0


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setattr('aso_agents.gui.subprocess.Popen', worker_process)
    jobs = Jobs(tmp_path)
    yield jobs
    for p in jobs.workers.values():p.returncode=0
    jobs.close()


def awaiting(jobs, questions=None):
    data = jobs.start({'request': 'Evaluate these ASOs using the supplied reference.'})
    data.update(status='needs_user_input', question={'id':'question1',
                'questions': questions or ['Which species?', 'Which cells?'], 'context': {}})
    write_json(jobs.directory(data['id']) / 'status.json', data)
    return data


def test_questions_are_dynamic_explicit_and_bound_to_the_current_round(jobs):
    data = awaiting(jobs)
    for payload in ({'question_id':'stale','answers':['human','neurons']},
                    {'question_id':'question1','answers':['human']},
                    {'question_id':'question1','answers':['human','']}):
        with pytest.raises(ValueError):
            jobs.answer(data['id'], payload)
    answer = {'question_id':'question1','answers':['human','neurons']}
    jobs.answer(data['id'], answer)
    assert read_json(jobs.directory(data['id']) / 'answer-question1.json') == [
        {'question':'Which species?','answer':'human'}, {'question':'Which cells?','answer':'neurons'}]
    with pytest.raises(ValueError):
        jobs.answer(data['id'], answer)
    with pytest.raises(ValueError):
        jobs.start({'request':'A second request'})


def test_worker_reuses_cli_lifecycle_options_answers_and_result(tmp_path, monkeypatch):
    write_json(tmp_path / 'request.json', {'request':'Design an ASO', 'output_dir':str(tmp_path),
        'skip_shortlist':True, 'state_path':str(tmp_path / 'state.json')})
    write_json(tmp_path / 'status.json', {'request':'Design an ASO'})
    def pipeline(**kwargs):
        assert kwargs['argv'] == ['--raw-cli','--skip-shortlist','--agentic-state-path',str(tmp_path/'state.json'),'--','Design an ASO']
        kwargs['event_callback']({'type':'progress','message':'Checking transcript structure'})
        answer_thread = threading.Thread(target=lambda: answer_when_ready(tmp_path))
        answer_thread.start()
        answers = kwargs['clarification_resolver'](questions=['Which cells?'], context={'reason':'Target context'})
        answer_thread.join(timeout=3)
        assert answers == [{'question':'Which cells?','answer':'neurons'}]
        kwargs['result_callback']({'status':'run_complete','recommendation_review':{
            'status':'unavailable','completion_notice':'Recommendation review remains incomplete.'}})
    monkeypatch.setenv('ASO_RUNS_DIR', str(tmp_path))
    run(tmp_path, pipeline=pipeline)
    data = read_json(tmp_path/'status.json')
    assert data['status'] == 'run_complete' and data['question'] is None
    assert data['conversation'][0]['answers'][0]['answer'] == 'neurons'
    assert data['result']['recommendation_review']['status'] == 'unavailable'
    assert data['progress'][-1]['message'] == 'Checking transcript structure'


def answer_when_ready(directory):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        data = read_json(directory/'status.json', {})
        if data.get('question'):
            question = data['question']
            write_json(directory/('answer-'+question['id']+'.json'), [{'question':q,'answer':'neurons'} for q in question['questions']])
            return
        time.sleep(.01)
    raise AssertionError('No clarification was emitted')


def test_worker_exception_does_not_copy_provider_secrets_to_browser(tmp_path, monkeypatch):
    write_json(tmp_path/'request.json', {'request':'Request','output_dir':str(tmp_path)})
    monkeypatch.setenv('ASO_RUNS_DIR', str(tmp_path))
    def fail(**kwargs):
        raise RuntimeError('sensitive-provider-token')
    run(tmp_path, pipeline=fail)
    saved = (tmp_path/'status.json').read_text()
    assert 'sensitive-provider-token' not in saved
    assert json.loads(saved)['status'] == 'fatal'


def test_output_scope_rejects_traversal_symlinks_and_unknown_runs(jobs, tmp_path):
    data = jobs.start({'request':'Request'})
    run_dir = tmp_path/'goal_1';run_dir.mkdir()
    (run_dir/'report.html').write_text('report')
    (run_dir/'reference.fa').symlink_to('/etc/passwd')
    data.update(run_dir=str(run_dir), result={'artifacts':{'report_html':str(run_dir/'report.html'),'genome':str(run_dir/'reference.fa')}})
    write_json(jobs.directory(data['id'])/'status.json', data)
    assert [f['name'] for f in jobs.snapshot(data['id'])['files']] == ['report.html']
    assert jobs.file(data['id'],'report.html').read_text() == 'report'
    for relative in ('../secret','/etc/passwd','reference.fa'):
        with pytest.raises(ValueError):
            jobs.file(data['id'], relative)
    with pytest.raises(ValueError):
        jobs.snapshot('../other')


def test_stopped_jobs_resume_their_recorded_state(jobs, tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr('aso_agents.gui.os.killpg', lambda *args:calls.append(args))
    data=awaiting(jobs, ['Confirm the reference?'])
    state=tmp_path/'run_state.json';state.write_text('{}')
    data['state_path']=str(state)
    write_json(jobs.directory(data['id'])/'status.json', data)
    jobs.stop(data['id'])
    assert jobs.snapshot(data['id'])['status']=='stopped' and len(calls)==2
    resumed=jobs.start({},resume=data['id'])
    saved=read_json(jobs.directory(resumed['id'])/'request.json')
    assert saved['state_path']==str(state) and saved['request']==data['request']


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr('aso_agents.gui.subprocess.Popen', worker_process)
    server=make_server(tmp_path,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield server
    # The fixture's workers have no operating-system process.
    for p in server.jobs.workers.values():p.returncode=0
    server.shutdown();server.jobs.close();server.server_close();thread.join()


def request(server,path,body=None,headers=None):
    headers={'Authorization':'Bearer '+server.token,**(headers or {})}
    if body is not None:headers['Content-Type']='application/json'
    req=urllib.request.Request('http://'+server.authority+path,
        data=json.dumps(body).encode() if body is not None else None,headers=headers)
    return urllib.request.urlopen(req,timeout=3)


def test_http_host_origin_auth_and_dynamic_answers(server):
    assert request(server,'/').status==200
    for headers in ({'Authorization':''},{'Origin':'https://another.example'},{'Host':'another.example'}):
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server,'/api/runs',headers=headers)
        assert error.value.code==403
    response=request(server,'/api/session')
    assert 'HttpOnly; SameSite=Strict' in response.headers['Set-Cookie']
    data=awaiting(server.jobs)
    assert json.load(request(server,'/api/runs/'+data['id']))['question']['questions']==['Which species?','Which cells?']
    assert request(server,'/api/runs/'+data['id']+'/answer',{'question_id':'question1','answers':['human','neurons']}).status==200


def test_cli_gui_routes_to_optional_host(monkeypatch, tmp_path):
    from aso_agents.cli import main
    calls=[]
    monkeypatch.setattr('aso_agents.gui.serve', lambda *a,**kw:calls.append((a,kw)) or 0)
    assert main(['gui','--no-open','--port','0','--output-dir',str(tmp_path)])==0
    assert calls==[((tmp_path,),{'port':0,'open_browser':False})]


def test_second_server_cannot_resume_another_servers_active_worker(jobs):
    with pytest.raises(ValueError):
        Jobs(jobs.output)


def test_browser_questions_progress_outputs_keyboard_and_mobile(server):
    import shutil
    playwright=pytest.importorskip('playwright.sync_api')
    executable=shutil.which('google-chrome') or shutil.which('chromium')
    if not executable:pytest.skip('Local browser unavailable')
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1280,'height':900})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://'+server.authority+'/#'+server.token)
        page.get_by_role('textbox',name='Your request',exact=True).fill('Compare supplied ASOs in neurons.')
        page.get_by_role('button',name='Start run',exact=True).click()
        page.wait_for_function("document.querySelector('#state').textContent==='Running'")
        identifier=server.jobs.list()[0]['id']
        data=server.jobs.snapshot(identifier)
        data.update(status='needs_user_input',question={'id':'browser-question',
            'questions':['Which organism?','Which delivery method?'], 'context':{'why_this_matters':'These inputs define the evaluation conditions.'}})
        write_json(server.jobs.directory(identifier)/'status.json',data)
        page.get_by_label('Which organism?').fill('human')
        page.get_by_label('Which delivery method?').fill('transfection')
        page.get_by_role('button',name='Continue',exact=True).click()
        page.wait_for_function("document.querySelector('#message').textContent.includes('Answer submitted')")
        answers=read_json(server.jobs.directory(identifier)/'answer-browser-question.json')
        assert [a['answer'] for a in answers]==['human','transfection']
        root=server.jobs.output/'goal_browser';root.mkdir()
        report=root/'report.html';report.write_text('<!doctype html><title>Result</title><h1>Saved result</h1>')
        csv=root/'candidates.csv';csv.write_text('ASO,score\nACGT,0.8\n')
        data.update(status='run_complete',question=None,run_dir=str(root),result={'status':'run_complete',
            'artifacts':{'report_html':str(report),'csv':str(csv)},'recommendation_review':{
                'status':'unavailable','completion_notice':'Recommendation review remains incomplete. 4 provisional ASOs are available in the report.'}})
        write_json(server.jobs.directory(identifier)/'status.json',data)
        page.get_by_role('link',name='Open report',exact=True).wait_for()
        assert '4 provisional ASOs' in page.locator('#completion').inner_text()
        with page.expect_popup() as popup:
            page.get_by_role('link',name='Open report',exact=True).click()
        assert popup.value.get_by_role('heading',name='Saved result').inner_text()=='Saved result'
        for width in (390,1280):
            page.set_viewport_size({'width':width,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth')<=width
        page.reload()
        page.get_by_role('button',name='Compare supplied ASOs in neurons.').click()
        page.get_by_role('link',name='Open report',exact=True).wait_for()
        assert not errors
        browser.close()
