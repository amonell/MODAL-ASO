"""Declared capabilities must reach the wire before a native run is admitted."""
import pytest
from agents import Agent, Runner, function_tool
from aso_agents.pipeline_controller_agent import pipeline_controller_agent
from aso_agents import pipeline_tools


def test_actual_controller_catalog_keeps_every_declared_capability(monkeypatch):
    observed=[]
    def capture(payload, **kwargs):
        observed.append(payload)
        return {'output':[{'type':'message','content':[{'type':'output_text','text':'{}'}]}]}
    monkeypatch.setattr(Runner,'_responses',staticmethod(capture))
    Runner._run_sync(pipeline_controller_agent,'Neutral registry diagnostic',max_steps=1)
    declared={getattr(tool,'_tool_name',tool.__name__) for tool in pipeline_controller_agent.tools}
    wire={tool['name'] for tool in observed[0]['tools']}
    assert wire==declared
    assert 'tool_run_steric_tiling' in wire
    assert not getattr(pipeline_tools._steric_tiling_physical_results_present,'_is_tool',False)


def test_unregistered_declared_tool_fails_before_any_provider_call(monkeypatch):
    def unregistered(value: str):return value
    monkeypatch.setattr(Runner,'_responses',lambda *a,**k:pytest.fail('must reject before provider call'))
    with pytest.raises(ValueError,match='unregistered'):
        Runner._run_sync(Agent(name='neutral',instructions='',tools=[unregistered]),'input')


def test_duplicate_tool_identity_cannot_shadow_a_different_callable(monkeypatch):
    @function_tool
    def first(value: str):return value
    @function_tool
    def second(value: str):return value
    second._tool_name=first._tool_name
    monkeypatch.setattr(Runner,'_responses',lambda *a,**k:pytest.fail('must reject before provider call'))
    with pytest.raises(ValueError,match='duplicate'):
        Runner._run_sync(Agent(name='neutral',instructions='',tools=[first,second]),'input')


def test_all_instantiated_production_agents_export_their_declared_tools():
    import ast
    import importlib
    from pathlib import Path
    checked=[]
    for path in sorted((Path(__file__).resolve().parents[1]/'aso_agents').glob('*.py')):
        tree=ast.parse(path.read_text())
        if not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                   and node.func.id=='Agent' for node in ast.walk(tree)):
            continue
        module=importlib.import_module('aso_agents.'+path.stem)
        for value in vars(module).values():
            if not isinstance(value,Agent) or not value.tools:continue
            names=[]
            for tool in value.tools:
                assert callable(tool) and getattr(tool,'_is_tool',False),(value.name,tool)
                names.append(tool._tool_name)
            assert len(names)==len(set(names)),value.name
            checked.append(value.name)
    assert pipeline_controller_agent.name in checked
    assert len(checked)>1
