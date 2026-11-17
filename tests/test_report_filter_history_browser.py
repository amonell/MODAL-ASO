"""Optional real-browser check of large exclusion inventories and CSV exports."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from aso_agents.readable_report import basic_document


@unittest.skipUnless(importlib.util.find_spec('playwright'), 'Optional Playwright browser checks')
class FilterHistoryBrowserTest(unittest.TestCase):
    def test_paging_search_stage_csv_safety_and_mobile(self):
        from playwright.sync_api import sync_playwright
        executable = os.environ.get('CHROME_PATH') or shutil.which('google-chrome') or shutil.which('chromium')
        if not executable:
            self.skipTest('Local Chrome/Chromium is required')
        with tempfile.TemporaryDirectory(prefix='aso-filter-report-') as temporary, sync_playwright() as p:
            root = Path(temporary)
            output = root/'artifacts/candidates/candidate_exclusions.json'
            output.parent.mkdir(parents=True)
            rows = [{'Synthesis':'ACGTACGT', 'IDT':'A*C*G*T*A*C*G*T', 'Chrom':'chr1', 'Start':i*10,
                     'End':i*10+8, 'GC':.123456789, 'stage':'Rule A' if i%2 else 'Rule B',
                     'reason':'=SUM(1,2)' if i%2 else '<img src=x onerror="window.unsafe=1">'} for i in range(10003)]
            output.write_text(json.dumps({'rows': rows}))
            document = basic_document('Filter history', '# Results\nNo eligible candidates.', root)
            browser = p.chromium.launch(executable_path=executable, headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'])
            page = browser.new_page(accept_downloads=True, viewport={'width':1280, 'height':900})
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.set_content(document)
            self.assertEqual(page.locator('.filter-history tbody tr').count(), 25)
            self.assertIn('10,003 records', page.locator('[data-filter=count]').inner_text())
            page.locator('[data-filter=size]').select_option('100')
            page.locator('[data-filter=next]').click()
            self.assertEqual(page.locator('.filter-history tbody tr').count(), 100)
            self.assertTrue(page.locator('[data-filter=count]').inner_text().startswith('101–200'))
            page.locator('[data-filter=stage]').select_option('Rule A')
            self.assertIn('5,001 records', page.locator('[data-filter=count]').inner_text())
            self.assertTrue(page.locator('[data-filter=previous]').is_disabled())
            with page.expect_download() as downloading:
                page.locator('[data-filter=csv]').click()
            destination = root/'filtered.csv'
            downloading.value.save_as(destination)
            exported = list(csv.DictReader(destination.open()))
            self.assertEqual(len(exported), 5001)
            self.assertTrue(all(r['reason']=="'=SUM(1,2)" and r['gc']=='0.123456789' for r in exported))
            page.locator('[data-filter=search]').fill('not a matching ASO')
            self.assertTrue(page.locator('[data-filter=csv]').is_disabled())
            self.assertEqual(page.locator('[data-filter=count]').inner_text(), '0 matching records')
            page.locator('[data-filter=search]').fill('ACGTACGT')
            self.assertIn('5,001 records', page.locator('[data-filter=count]').inner_text())
            page.set_viewport_size({'width':390, 'height':844})
            self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), 390)
            self.assertIsNone(page.evaluate('window.unsafe'))
            self.assertEqual(errors, [])
            browser.close()
