"""Warnings stay in optional technical details without hiding run failures."""
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest import TestCase, SkipTest
from unittest.mock import patch
import sys

try:
    from streamlit.testing.v1 import AppTest
except ImportError as exc:
    raise SkipTest('Optional GUI dependencies are not installed') from exc

from aso_agents.gui.config import Settings
from aso_agents.gui.worker import ChatStream

WARNING = (
    '/library/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:\n'
    '  PydanticSerializationUnexpectedValue(Expected Message, got secret-key)\n'
    '  PydanticSerializationUnexpectedValue(Expected another field)\n'
    '  return self.__pydantic_serializer__.to_python(\n'
)


class WarningTests(TestCase):
    def test_multiline_warnings_are_grouped_and_redacted(self):
        events = Queue()
        stream = ChatStream(events, 'secret-key')
        stream.write('Starting analysis.\n')
        for offset in range(0, len(WARNING), 13):
            stream.write(WARNING[offset:offset + 13])
        stream.write(WARNING)
        stream.write('Analysis continues.\n')
        stream.finish()
        output = []
        while not events.empty():
            output.append(events.get_nowait())
        self.assertEqual([kind for kind, _ in output], ['message', 'diagnostic', 'diagnostic', 'message'])
        self.assertEqual(output[-1], ('message', 'Analysis continues.'))
        self.assertNotIn('secret-key', str(output))
        self.assertIn('return self.__pydantic_serializer__', output[1][1]['traceback'])

    def test_unterminated_warning_flush_does_not_swallow_error(self):
        events = Queue()
        stream = ChatStream(events, '')
        stream.write('Pydantic serializer warnings:\n  unexpected field\n')
        stream.write('Error: provider request failed\n')
        stream.write('PydanticSerializationUnexpectedValue: final warning')
        stream.finish()
        self.assertEqual(events.get_nowait()[0], 'diagnostic')
        self.assertEqual(events.get_nowait(), ('message', 'Error: provider request failed'))
        self.assertEqual(events.get_nowait()[0], 'diagnostic')
        self.assertTrue(events.empty())

    def test_warning_summary_is_opt_in_but_failure_stays_visible(self):
        events, answers = Queue(), Queue()
        stream = ChatStream(events, 'secret-key')
        stream.write(WARNING * 3)
        stream.write('Analysis continues.\n')
        stream.finish()
        events.put(('error', 'Analysis stopped: provider request failed.'))
        events.put(('diagnostic', {'type': 'RuntimeError', 'message': 'provider request failed',
                                   'traceback': 'RuntimeError: provider request failed'}))
        events.put(('done', None))
        with patch.object(sys, 'argv', ['app.py', '--test']):
            ui = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'aso_agents/gui/app.py'))
            ui.session_state['validated'] = Settings('openai', 'correctkey')
            ui.session_state['messages'] = []
            ui.session_state['job'] = (SimpleNamespace(is_alive=lambda: True, exitcode=None), events, answers)
            ui.run(timeout=20)
            self.assertFalse(ui.exception)
            self.assertFalse(any('SDK serialization' in item.label for item in ui.expander))
            self.assertFalse(any('pydantic_serializer' in item.value for item in ui.markdown))
            self.assertTrue(any('Analysis stopped' in item.value for item in ui.markdown))
            self.assertTrue(any('RuntimeError' in item.label for item in ui.expander))
            warnings = [message for role, message in ui.session_state['messages']
                        if role == 'diagnostic' and message['type'] == 'SDK serialization warning']
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0]['count'], 3)
            self.assertEqual(len(warnings[0]['samples']), 1)
            ui.checkbox[0].check().run()
            self.assertFalse(ui.exception)
            self.assertEqual(sum('SDK serialization' in item.label for item in ui.expander), 1)
            self.assertTrue(any('3 warning(s)' in item.value for item in ui.caption))
            ui.checkbox[0].uncheck().run()
            self.assertFalse(any('SDK serialization' in item.label for item in ui.expander))
