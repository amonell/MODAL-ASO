"""Offline GUI fixtures, enabled only by the explicit --test app argument."""
from pathlib import Path
import gzip
import json
import shutil
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Thread
from time import sleep

from aso_agents.gui.config import DEFAULT_OPENROUTER_MODEL, SetupError

PREVIEW_ROOT = Path(__file__).resolve().parents[1] / 'resources' / 'gui' / 'test_run'


def prepare_outputs(root):
    """Copy the frozen run snapshot; never invoke analysis or modify live runs."""
    metadata = json.loads((PREVIEW_ROOT / 'preview.json').read_text(encoding='utf-8'))
    for item in metadata['files']:
        relative = Path(item['path'])
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.suffix == '.html':
            with gzip.open(PREVIEW_ROOT / (str(relative) + '.gz'), 'rb') as source, destination.open('wb') as output:
                shutil.copyfileobj(source, output)
        else:
            shutil.copyfile(PREVIEW_ROOT / relative, destination)
    return metadata


def validate(settings, models=None):
    if settings.api_key != "correctkey":
        raise SetupError("API key", "Authentication failed. Use correctkey in test mode.")


def list_models(provider, key):
    if key != "correctkey":
        raise SetupError("API key", "Authentication failed. Use correctkey in test mode.")
    return {DEFAULT_OPENROUTER_MODEL: ["low", "medium", "high"]}


def configuration_preview(settings):
    return {"Provider": settings.provider, "Model": "Offline mock model", "Reasoning effort": settings.effort}


def start(settings, request):
    validate(settings)
    events, answers = Queue(), Queue()
    # Keep demo files alive for the session's download callbacks.
    directory = TemporaryDirectory(prefix="modal-aso-ui-")
    def respond():
        events.put(("message", "Test request received. I'll load the saved PCSK9 report and files for this interface preview. No new analysis will run."))
        for message in ["Mapping transcripts and target sites.", "Checking candidate sites.", "Mapping transcripts and target sites."]:
            sleep(2)
            events.put(("message", message))
            events.put(("activity", message))
        root = Path(directory.name)
        metadata = prepare_outputs(root)
        events.put(("result", {"status": "run_complete", "run_dir": str(root),
                               "preview_source": metadata['source_run']}))
        events.put(("done", None))
    thread = Thread(target=respond, daemon=True)
    thread.exitcode = None
    thread.output_directory = directory
    thread.start()
    return thread, events, answers
