"""MODAL-ASO design and evaluation workflows; agents load on demand."""
from importlib import import_module
import sys
from types import ModuleType

__version__ = "0.1.0"

_EXPORTS = {'goal_intake_agent': 'goal_intake_agent', 'extract_transcript_lock': 'goal_intake_agent', 'transcript_chooser_agent': 'transcript_chooser_agent', 'chemistry_agent': 'chemistry_agent', 'markdown_agent': 'markdown_agent', 'report_explainer_agent': 'report_explainer_agent', 'splicing_agent': 'splicing_agent', 'generate_splice_strategy_report': 'splice_strategy_agent', 'resolve_ambiguous_safeguards': 'splice_strategy_agent', 'resolve_mixed_junction_directions': 'splice_strategy_agent', 'splice_strategy_router_agent': 'splice_strategy_router_agent', 'splice_strategy_router_planner_agent': 'splice_strategy_router_agent', 'execute_router_plan': 'splice_strategy_router_agent', 'off_target_expression_agent': 'off_target_expression_agent'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


class _AgentExports(ModuleType):
    """Keep legacy agent exports stable after their submodules are imported."""

    def __getattribute__(self, name):
        if name in _EXPORTS:
            return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _AgentExports
