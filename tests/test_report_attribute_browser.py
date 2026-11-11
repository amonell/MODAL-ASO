"""Optional real-browser regression for the offline attribute explorer.

Run with a Python environment containing Playwright and local Chrome/Chromium:
python -m unittest discover -s tests -p test_report_attribute_browser.py
CHROME_PATH can select the executable; TMPDIR selects the artifact filesystem.
"""
import csv
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from aso_agents.readable_report import basic_document
from aso_agents.report_explorer_data import project_records


@unittest.skipUnless(importlib.util.find_spec('playwright'), 'Optional Playwright browser checks')
class AttributeBrowserTest(unittest.TestCase):
    def test_color_sliders_intersection_missing_export_and_mobile(self):
        from playwright.sync_api import sync_playwright
        executable = os.environ.get('CHROME_PATH') or shutil.which('google-chrome') or shutil.which('chromium')
        if not executable:
            self.skipTest('Local Chrome/Chromium is required')
        rows = [{'name': f'ASO {i}', 'antisense_5to3': ['ACGT', 'CGTA', 'GTAC', 'TACG'][i],
                 'ordering_notation': ['ACGT', 'CGTA', 'GTAC', 'TACG'][i],
                 'binding_intervals_bed0_half_open': [['chr1', 100 + i * 5, 104 + i * 5]],
                 'novel_measurement': [0, 10, 20, None][i], 'novel_batch': ['A', 'B', 'B', 'A'][i],
                 'special_label': ['constructor', '__proto__', 'constructor', '__proto__'][i],
                 'custom_tag': f'tag-{i}', 'candidate_number': i + 1,
                 'IDT': ['Order A', 'Order B', 'Order B', 'Order A'][i]}
                for i in range(4)]
        data = project_records(rows)
        attributes = {a['label']: a['id'] for a in data['attributes']}
        names = {r['id']: r['name'] for r in data['records']}
        document = basic_document('Attribute regression', '# Results\nSynthetic interaction checks.', candidate_data=data)
        with tempfile.TemporaryDirectory(prefix='aso-attributes-') as temporary, sync_playwright() as p:
            browser = p.chromium.launch(executable_path=executable, headless=True,
                                        args=['--no-sandbox', '--disable-dev-shm-usage'])
            page = browser.new_page(accept_downloads=True, viewport={'width': 1280, 'height': 900})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.set_content(document)

            def choose(label, action='color'):
                page.locator(f'[data-attribute-action={action}]').click()
                page.locator('[data-role=attribute-search]').fill(label)
                self.assertTrue(page.locator('[data-attribute-action=choose]').is_enabled())
                page.locator('[data-role=attribute-search]').press('Enter')

            def shown():
                ids = page.locator('tbody tr[data-record]').evaluate_all('(rows)=>rows.map(r=>r.dataset.record)')
                return {names[key] for key in ids}

            page.locator('[data-attribute-action=color]').click()
            for label in ('IDT', 'Custom tag', 'Candidate number'):
                page.locator('[data-role=attribute-search]').fill(label)
                self.assertEqual(page.locator('[data-role=attribute-options] option').count(), 0, label)
                self.assertFalse(page.locator('[data-attribute-action=choose]').is_enabled())
            page.locator('[data-attribute-action=close]').click()
            choose('IDT', 'add')
            self.assertTrue(page.locator('[data-filter-id="' + attributes['IDT'] + '"]').is_visible())
            page.locator('[data-attribute-action=clear]').click()

            choose('Novel measurement')
            colors = page.locator('.report-attribute-swatch').evaluate_all('(els)=>els.map(e=>e.style.backgroundColor)')
            self.assertEqual(len(set(colors)), 4)  # Zero and unavailable have different colors.
            self.assertIn('3 ASOs with values', page.locator('[data-role=attribute-color-histogram]').inner_text())
            choose('Novel measurement', 'add')
            card = page.locator('[data-filter-id="' + attributes['Novel measurement'] + '"]')
            self.assertEqual(card.locator('[data-filter-input=low-slider]').get_attribute('step'), '1')
            card.locator('[data-filter-input=low-slider]').evaluate(
                '(el)=>{el.value=10;el.dispatchEvent(new Event("input",{bubbles:true}));}')
            self.assertEqual(shown(), {'ASO 1', 'ASO 2'})
            self.assertEqual(card.locator('[data-filter-input=low]').input_value(), '10')
            choose('Novel batch', 'add')
            category = page.locator('[data-filter-id="' + attributes['Novel batch'] + '"]')
            category.locator('[data-filter-category="A"]').check()
            self.assertEqual(shown(), set())
            self.assertTrue(page.locator('[data-role=map-section]').is_visible())
            card.locator('[data-filter-input=missing]').check()
            self.assertEqual(shown(), {'ASO 3'})
            self.assertEqual(page.locator('[data-attribute-column]').count(), 2)
            with page.expect_download() as downloaded:
                page.locator('[data-action=filtered-csv]').click()
            path = Path(temporary) / 'matching.csv'
            downloaded.value.save_as(path)
            with path.open() as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual([r['name'] for r in exported], ['ASO 3'])
            self.assertEqual(exported[0]['Attribute: Novel batch'], 'A')
            page.locator('[data-attribute-action=clear]').click()
            self.assertEqual(shown(), set(names.values()))
            choose('Special label')
            page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
            self.assertIn('constructor', page.locator('[data-role=category-legend]').inner_text())
            self.assertNotIn('[object', page.locator('tbody').inner_text())
            page.set_viewport_size({'width': 390, 'height': 844})
            self.assertFalse(page.evaluate('document.documentElement.scrollWidth>innerWidth+1'))
            self.assertEqual(errors, [])
            browser.close()
