"""Browser checks for recommendation reading, selection and large panels."""
import csv
import importlib.util
import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from aso_agents.readable_report import basic_document
from aso_agents.report_explorer_data import project_records


def recommendation_data(count=12):
    rows = []
    for i in range(count):
        sequence = 'ACGTACGT' + ''.join('ACGT'[(i//4**j)%4] for j in range(5))
        rows.append({'name': 'ASO ' + str(i+1), 'antisense_5to3': sequence,
            'ordering_notation': sequence, 'binding_intervals_bed0_half_open': [['chr1', 100+i*20, 113+i*20]]})
    data = project_records(rows, kind='design')
    data['recommendation_review'] = {'status': 'completed', 'reviewed_molecule_count': count,
        'summary': 'The panel samples distinct recorded binding sites.',
        'size_reason': 'The count follows the request.', 'reviewer': {'model': 'fixture-model'}}
    for i, row in enumerate(data['records'], 1):
        row['panel_rank'] = i
        row['recommendation'] = {'candidate_id': 'fixture_'+str(i),
            'why_chosen': f'ASO {i} covers a recorded binding site.',
            'panel_role': f'Adds binding location {i}.',
            'tradeoff': 'Target-site accessibility remains unresolved.',
            'evidence': [{'candidate_id': 'fixture_'+str(i), 'path': '/contexts/0/recorded_observation',
                          'value': '<script>unsafe()</script>'}]}
        row['selection_note'] = row['recommendation']['why_chosen']
    return data


@unittest.skipUnless(importlib.util.find_spec('playwright'), 'Optional Playwright browser checks')
class RecommendationBrowserTest(unittest.TestCase):
    def test_selection_feedback_basket_filter_undo_and_keyboard(self):
        from playwright.sync_api import sync_playwright
        executable = os.environ.get('CHROME_PATH') or shutil.which('google-chrome') or shutil.which('chromium')
        if not executable:
            self.skipTest('Local Chrome/Chromium is required')
        data = recommendation_data(6)
        for row in data['records'][4:]:
            row['panel_rank'] = None
            row['recommendation'] = {}
        document = basic_document('Selection UX checks', '', candidate_data=data)
        with tempfile.TemporaryDirectory(prefix='aso-selection-ux-') as temporary, sync_playwright() as p:
            source = Path(temporary)/'report.html'
            source.write_text(document)
            browser = p.chromium.launch(executable_path=executable, headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'])
            page = browser.new_page(accept_downloads=True, viewport={'width': 1280, 'height': 900})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(source.as_uri())
            first = page.locator('[data-recommendation-pick]').first
            first.click()
            self.assertEqual(first.inner_text(), 'Remove ASO 1 from basket')
            self.assertEqual(first.get_attribute('data-selected'), 'true')
            first.click()
            self.assertEqual(first.inner_text(), 'Add ASO 1 to basket')
            self.assertEqual(page.locator('[data-role=basket-count]').inner_text(), 'No ASOs selected')

            # Rerendering the table must keep keyboard users at their checkbox.
            checkbox = page.locator('[data-pick]').first
            checkbox.focus()
            checkbox.press('Space')
            self.assertTrue(checkbox.is_checked())
            self.assertTrue(checkbox.evaluate('(el)=>el===document.activeElement'))
            checkbox.press('Space')
            self.assertFalse(checkbox.is_checked())
            self.assertTrue(checkbox.evaluate('(el)=>el===document.activeElement'))

            page.locator('[data-action=add-recommendations]').click()
            self.assertTrue(page.locator('[data-action=add-panel]').is_disabled())
            self.assertEqual(page.locator('[data-action=add-panel]').inner_text(), 'Recommended ASOs added')
            page.get_by_label('In my basket', exact=True).check()
            self.assertEqual(page.locator('tbody tr[data-record]').count(), 4)
            checkbox = page.locator('[data-pick]').first
            checkbox.focus()
            checkbox.press('Space')
            self.assertEqual(page.locator('tbody tr[data-record]').count(), 3)
            self.assertTrue(page.get_by_role('region', name='ASO candidate table', exact=True).evaluate('(el)=>el===document.activeElement'))
            first.click()
            self.assertEqual(page.locator('tbody tr[data-record]').count(), 4)

            page.locator('[data-action=clear]').click()
            undo = page.locator('[data-action=undo-clear]')
            self.assertTrue(undo.is_visible())
            self.assertTrue(undo.evaluate('(el)=>el===document.activeElement'))
            self.assertEqual(page.locator('tbody tr[data-record]').count(), 0)
            page.locator('[data-action=empty-reset]').click()
            self.assertEqual(page.locator('tbody tr[data-record]').count(), 6)
            self.assertTrue(page.locator('[data-role=search]').evaluate('(el)=>el===document.activeElement'))
            undo.click()
            self.assertEqual(page.locator('[data-role=basket-count]').inner_text(), '4 ASOs selected')
            self.assertFalse(undo.is_visible())

            # The basket can include ASOs outside the recommended panel.
            extra = data['records'][4]
            page.locator('[data-pick="'+extra['id']+'"]').check()
            self.assertEqual(page.locator('[data-role=basket-count]').inner_text(), '5 ASOs selected')
            page.get_by_label('Recommended ASOs only', exact=True).check()
            page.get_by_label('In my basket', exact=True).check()
            self.assertEqual(page.locator('tbody tr[data-record]').count(), 4)
            first.click()
            self.assertEqual(page.locator('tbody tr[data-record]').count(), 3)
            page.reload()
            self.assertEqual(page.locator('[data-role=basket-count]').inner_text(), '4 ASOs selected')
            self.assertEqual(first.inner_text(), 'Add ASO 1 to basket')
            with page.expect_download() as downloading:
                page.locator('[data-action=csv]').click()
            saved = Path(temporary)/'selected.csv'
            downloading.value.save_as(saved)
            exported = list(csv.DictReader(io.StringIO(saved.read_text())))
            self.assertEqual([row['antisense_5to3'] for row in exported],
                             [row['aso'] for row in data['records'][1:5]])
            page.locator('[data-action=show-recommendations]').press('Enter')
            self.assertTrue(page.get_by_role('region', name='ASO candidate table', exact=True).evaluate('(el)=>el===document.activeElement'))
            self.assertEqual(page.locator('tbody tr[data-record]').count(), 4)
            page.set_viewport_size({'width': 390, 'height': 844})
            self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), 390)
            self.assertEqual(errors, [])
            browser.close()

    def test_shortlist_disclosure_basket_export_and_mobile(self):
        from playwright.sync_api import sync_playwright
        executable = os.environ.get('CHROME_PATH') or shutil.which('google-chrome') or shutil.which('chromium')
        if not executable:
            self.skipTest('Local Chrome/Chromium is required')
        data = recommendation_data()
        data['records'].reverse()
        document = basic_document('Recommendation checks', '', candidate_data=data)
        with tempfile.TemporaryDirectory(prefix='aso-recommendations-') as temporary, sync_playwright() as p:
            browser = p.chromium.launch(executable_path=executable, headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'])
            page = browser.new_page(accept_downloads=True, viewport={'width': 1280, 'height': 900})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.set_content(document)
            self.assertEqual(page.locator('.report-recommendation-list>li').count(), 8)
            self.assertIn('all 12 ASOs', page.locator('[data-role=recommendations]').inner_text())
            first = page.locator('.report-recommendation-list>li').first
            first.get_by_text('Why this ASO belongs in the panel', exact=True).click()
            self.assertTrue(first.get_by_text('Target-site accessibility remains unresolved.', exact=False).is_visible())
            first.get_by_text('Evidence behind this choice', exact=True).click()
            self.assertIn('<script>unsafe()</script>', first.inner_text())
            page.locator('[data-action=add-recommendations]').click()
            self.assertIn('12 ASOs', page.locator('[data-role=basket-count]').inner_text())
            with page.expect_download() as downloading:
                page.locator('[data-action=csv]').click()
            saved = Path(temporary)/'panel.csv'
            downloading.value.save_as(saved)
            exported = list(csv.DictReader(io.StringIO(saved.read_text())))
            self.assertEqual(len(exported), 12)
            self.assertTrue(all(row['recommendation_tradeoff'] for row in exported))
            self.assertEqual([int(row['panel_order']) for row in exported], list(range(1, 13)))
            page.locator('[data-action=show-recommendations]').click()
            self.assertTrue(page.locator('[data-role=panel-only]').is_checked())
            last = page.locator('[data-role=candidate-table] tbody tr[data-record]').last
            last.locator('summary').click()
            self.assertIn('Adds binding location 12.', last.inner_text())
            page.set_viewport_size({'width': 390, 'height': 844})
            self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), 390)
            self.assertEqual(errors, [])
            offline = browser.new_page(java_script_enabled=False)
            offline.set_content(document)
            self.assertEqual(offline.locator('.report-recommendation-list>li').count(), 8)
            browser.close()
