"""Offline GUI smoke checks; runnable with unittest without pytest."""
from pathlib import Path
from unittest import TestCase, SkipTest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import hashlib
import json
import sys

try:
    from streamlit.testing.v1 import AppTest
except ImportError as exc:
    raise SkipTest('Optional Streamlit GUI dependencies are not installed') from exc
from aso_agents.gui.config import Settings, SetupError
from aso_agents.gui import mock

APP = Path(__file__).resolve().parents[1] / 'aso_agents/gui/app.py'


class MockGuiTests(TestCase):
    def test_keys_and_offline_chat(self):
        with patch.object(sys, 'argv', ['app.py', '--test']), \
                patch('aso_agents.gui.config.list_models', side_effect=AssertionError('Live API')), \
                patch('aso_agents.gui.worker.start', side_effect=AssertionError('Real worker')):
            ui = AppTest.from_file(str(APP)).run(timeout=20)
            ui.text_input[0].set_value('wrongkey').run()
            next(button for button in ui.button if button.label == 'Start Chat').click().run()
            self.assertTrue(any('Authentication failed' in error.value for error in ui.error))
            self.assertFalse(ui.chat_input)
            ui.text_input[0].set_value('correctkey').run()
            next(button for button in ui.button if button.label == 'Start Chat').click().run()
            self.assertFalse(ui.exception)
            self.assertEqual([tab.label for tab in ui.tabs], ['Chat', 'Results', 'Files'])
            # A fresh test tree avoids stale setup widgets retained across rerun
            # by Streamlit's AppTest implementation.
            settings = ui.session_state['validated']
            ui = AppTest.from_file(str(APP))
            ui.session_state['validated'] = settings
            ui.session_state['messages'] = []
            ui.run()
            ui.chat_input[0].set_value('Preview my request').run()
            self.assertIn(('user', 'Preview my request'), ui.session_state['messages'])
            self.assertTrue(any(item.value == 'Preview my request' for item in ui.markdown))
            process = ui.session_state['job'][0]
            process.join(timeout=10)
            ui.run()
            self.assertFalse(ui.exception)
            self.assertEqual(ui.session_state['messages'].count(('assistant', 'Mapping transcripts and target sites.')), 1)
            self.assertEqual(len(ui.get('download_button')), 6)
            self.assertTrue(ui.get('iframe'))
            self.assertEqual(ui.session_state['result']['preview_source'], 'goal_1794942511_PCSK9')
            self.assertTrue(any('/_modal_aso_artifacts/' in item.proto.srcdoc for item in ui.get('iframe')))
            self.assertTrue(any('recommended_shortlist.csv' in item.value for item in ui.markdown))
            ui.chat_input[0].set_value('Second request').run()
            process = ui.session_state['job'][0]
            process.join(timeout=10)
            ui.run()
            self.assertEqual(ui.session_state['messages'].count(('assistant', 'Mapping transcripts and target sites.')), 2)
            self.assertFalse(ui.exception)
            next(button for button in ui.button if button.label == 'View Results').click().run()
            self.assertEqual(ui.session_state['workspace_tab'], 'Results')
            next(button for button in ui.button if button.label == 'View Files').click().run()
            self.assertEqual(ui.session_state['workspace_tab'], 'Files')
            self.assertFalse(ui.exception)

    def test_mock_rejects_other_keys(self):
        for key in ['', 'wrongkey', 'CORRECTKEY']:
            with self.assertRaises(SetupError):
                mock.validate(Settings('openai', key))
        mock.validate(Settings('openai', 'correctkey'))

    def test_saved_preview_outputs_are_exact_copies(self):
        metadata = json.loads((mock.PREVIEW_ROOT / 'preview.json').read_text(encoding='utf-8'))
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in mock.PREVIEW_ROOT.rglob('*') if path.is_file()}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(mock.prepare_outputs(root), metadata)
            for item in metadata['files']:
                content = (root / item['path']).read_bytes()
                self.assertEqual(len(content), item['bytes'])
                self.assertEqual(hashlib.sha256(content).hexdigest(), item['sha256'])
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})

    def test_launcher_forwards_test_as_app_argument(self):
        from aso_agents.gui.launcher import launch
        with patch('aso_agents.gui.launcher.subprocess.call', return_value=0) as call:
            launch(['--test', '--server.port=8502'])
        self.assertIn('--server.fileWatcherType=none', call.call_args.args[0])
        self.assertIn('--global.developmentMode=false', call.call_args.args[0])
        arguments = call.call_args.args[0]
        self.assertEqual(arguments[-2:], ['--', '--test'])
        self.assertIn('--server.port=8502', arguments)
