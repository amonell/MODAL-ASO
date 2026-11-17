"""Optional real-browser checks for dense maps and all preserved ASO interactions."""
from copy import deepcopy
import importlib.util
import os
import shutil
import unittest

from aso_agents.readable_report import basic_document
from aso_agents.report_explorer_data import project_records


# Observe the actual rendered geometry, including the browser's font metrics.
# Each background repaint starts a fresh frame; no application test hooks needed.
DRAWING_PROBE = r'''()=>{
 const proto=CanvasRenderingContext2D.prototype,fill=proto.fillRect,text=proto.fillText;
 window.mapDrawing={labels:[],bars:[]};
 proto.fillRect=function(x,y,w,h){
  if(this.canvas.dataset.role==='map'){
   if(x===0&&y===0)window.mapDrawing={labels:[],bars:[]};
   if(h===7)window.mapDrawing.bars.push({x,y,w,h});
  }
  return fill.apply(this,arguments);
 };
 proto.fillText=function(value,x,y){
  if(this.canvas.dataset.role==='map'){
   const m=this.measureText(value),left=x-(this.textAlign==='right'?m.width:this.textAlign==='center'?m.width/2:0);
   window.mapDrawing.labels.push({text:String(value),left,right:left+m.width,top:y-m.actualBoundingBoxAscent,bottom:y+m.actualBoundingBoxDescent});
  }
  return text.apply(this,arguments);
 };
}'''


def fixture(regions=2, per_region=120):
    rows=[]
    for region in range(regions):
        for i in range(per_region):
            index=region*per_region+i
            sequence='ACGT'*3+''.join('ACGT'[(index//4**j)%4] for j in range(8))
            rows.append({'name':f'ASO {index+1}','antisense_5to3':sequence,
                'ordering_notation':sequence,'novel_measurement':index,
                'novel_batch':'A' if i%2 else 'B',
                'binding_intervals_bed0_half_open':[['chr1',100000+region*10000+i,100020+region*10000+i]]})
    rows.append({'name':'Across two regions','antisense_5to3':'T'*20,'ordering_notation':'T'*20,
        'novel_measurement':len(rows),'novel_batch':'B',
        'binding_intervals_bed0_half_open':[['chr1',100008,100018],['chr1',110008,110018]]})
    data=project_records(rows)
    data['contexts'][0].update(reference_chrom='chr1',reference_strand='-',transcript='Saved transcript',
        exons=[{'start':100015,'end':100110,'label':'Annotated exon 2','index':2}])
    data['contexts'][0]['sites']=[{'chrom':'chr1','position':100005+i*5,
        'target':i==9,'label':f'Donor at {100005+i*5:,}'} for i in range(20)]
    data['contexts'][0]['sites'].append({'chrom':'chr1','position':999999,'label':'Distant recorded site','target':False})
    return data


@unittest.skipUnless(importlib.util.find_spec('playwright'),'Optional Playwright browser checks')
class BindingMapBrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        executable=os.environ.get('CHROME_PATH') or shutil.which('google-chrome') or shutil.which('chromium')
        if not executable:raise unittest.SkipTest('Local Chrome/Chromium is required')
        cls.playwright=sync_playwright().start()
        cls.browser=cls.playwright.chromium.launch(executable_path=executable,headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage'])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.playwright.stop()

    def setUp(self):
        self.page=self.browser.new_page(viewport={'width':1440,'height':1000},reduced_motion='reduce')
        self.errors=[]
        self.page.on('pageerror',lambda error:self.errors.append(str(error)))
        self.page.evaluate(DRAWING_PROBE)

    def tearDown(self):
        self.page.close()
        self.assertEqual(self.errors,[])

    def settle(self):
        self.page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')

    def choose_color(self,label):
        self.page.locator('[data-attribute-action=color]').click()
        self.page.locator('[data-role=attribute-search]').fill(label)
        self.page.locator('[data-role=attribute-search]').press('Enter')
        self.settle()

    def assert_clear_labels(self):
        labels=self.page.evaluate('window.mapDrawing.labels')
        self.assertTrue(labels)
        width=self.page.locator('[data-role=map]').bounding_box()['width']
        for i,label in enumerate(labels):
            self.assertGreaterEqual(label['left'],-.5,label)
            self.assertLessEqual(label['right'],width+.5,label)
            for other in labels[i+1:]:
                intersect=(label['left']<other['right'] and other['left']<label['right'] and
                           label['top']<other['bottom'] and other['top']<label['bottom'])
                self.assertFalse(intersect,(label,other))

    def test_dense_map_paging_regions_values_filters_selection_and_mobile(self):
        data=fixture();before=deepcopy(data)
        self.page.set_content(basic_document('Dense binding map','',candidate_data=data));self.settle()
        self.assertEqual(data,before)
        self.assertEqual(self.page.locator('[data-role=map-region] option').count(),3)
        self.assertIn('20 in view · 21 recorded',self.page.locator('[data-role=map-sites-summary]').inner_text())
        self.assert_clear_labels()
        self.assertTrue(any('Target' in label['text'] for label in self.page.evaluate('window.mapDrawing.labels')))
        self.assertFalse(self.page.locator('[data-role=map-values]').is_checked())

        # Every candidate is reachable through map pagination, including a
        # junction-spanning ASO whose two blocks occupy one row together.
        seen=0
        while True:
            status=self.page.locator('[data-role=map-status]').inner_text()
            seen+=int(status.split()[1].replace(',',''))
            if self.page.locator('[data-action=map-next]').is_disabled():break
            self.page.locator('[data-action=map-next]').click();self.settle()
        self.assertEqual(seen,len(data['records']))
        table_ids=self.page.locator('[data-record]').evaluate_all('(els)=>els.map(e=>e.dataset.record)')
        self.page.locator('[data-role=map-region]').select_option('1');self.settle()
        self.assertIn('of 121 matching ASOs',self.page.locator('[data-role=map-status]').inner_text())
        self.assertIn('Region 1',self.page.locator('[data-role=map-target-note]').inner_text())
        self.assertEqual(table_ids,self.page.locator('[data-record]').evaluate_all('(els)=>els.map(e=>e.dataset.record)'))
        self.page.locator('[data-role=map-sites] summary').click()
        self.assertEqual(self.page.locator('[data-role=map-sites-body] tr').count(),21)
        target=self.page.locator('[data-role=map-sites-body] tr').filter(has_text='Target')
        target.get_by_role('button',name='Region 1').click();self.settle()
        self.assertEqual(self.page.locator('[data-role=map-region]').input_value(),'0')
        self.assertTrue(self.page.locator('[data-role=map-target-note]').is_hidden())
        self.assertTrue(target.get_by_role('button').evaluate('(el)=>el===document.activeElement'))
        self.page.locator('[data-role=map-sites] summary').click()

        self.choose_color('Novel measurement')
        self.page.locator('[data-role=map-values]').check();self.settle()
        self.assert_clear_labels()
        self.assertIn('Novel measurement',[label['text'] for label in self.page.evaluate('window.mapDrawing.labels')])
        self.choose_color('Novel batch')
        self.assertTrue(self.page.locator('[data-role=map-values-control]').is_hidden())
        self.assert_clear_labels()
        self.page.locator('[data-role=map-region]').select_option('all');self.settle()
        canvas=self.page.locator('[data-role=map]')
        bar=self.page.evaluate('window.mapDrawing.bars[0]')
        canvas.click(position={'x':bar['x']+bar['w']/2,'y':bar['y']+3});self.settle()
        self.assertIn('1 ASO',self.page.locator('[data-role=basket-count]').inner_text())
        self.page.locator('[data-role=search]').fill('no matching sequence');self.settle()
        self.assertIn('Showing 0 of 0',self.page.locator('[data-role=map-status]').inner_text())
        self.assertTrue(self.page.locator('[data-role=map-paging]').is_hidden())
        self.assert_clear_labels()
        self.page.locator('[data-role=search]').fill('');self.settle()
        for width in [390,1101,1280]:
            self.page.set_viewport_size({'width':width,'height':900});self.settle()
            self.assertFalse(self.page.evaluate('document.documentElement.scrollWidth>innerWidth+1'))
            self.assertTrue(self.page.locator('.report-map').evaluate('(el)=>el.scrollWidth<=el.clientWidth+1'))
            self.assert_clear_labels()

    def test_large_catalog_defaults_to_target_and_switches_chromosome_and_context(self):
        data=fixture(regions=8,per_region=5)
        context=data['contexts'][0]
        context['sites']=[{'chrom':'chr1','position':150010,'label':'Requested target','target':True},
                          {'chrom':'chr2','position':170010,'label':'Second chromosome target','target':True}]
        data['records'][-2]['intervals'][0]['chrom']='chr2'
        other={**deepcopy(context),'id':'other','label':'Second approach','sites':[]}
        data['contexts'].append(other)
        data['records'][1]['context']='other'
        self.page.set_content(basic_document('Many regions','',candidate_data=data));self.settle()
        self.assertEqual(self.page.locator('[data-role=map-region]').input_value(),'5')
        self.assertIn('Region 6 of 8',self.page.locator('[data-role=map-description]').inner_text())
        self.assert_clear_labels()
        self.page.locator('[data-role=map-region]').select_option('all');self.settle()
        self.assertFalse(self.page.evaluate('document.documentElement.scrollWidth>innerWidth+1'))
        self.assert_clear_labels()
        self.page.locator('[data-role=map-chromosome]').select_option('2');self.settle()
        self.assertIn('Chromosome 2',self.page.locator('[data-role=map-description]').inner_text())
        self.assertIn('Showing 1 of 1',self.page.locator('[data-role=map-status]').inner_text())
        self.assertTrue(self.page.locator('[data-role=map-region-control]').is_hidden())
        self.page.locator('[data-role=context]').select_option('other');self.settle()
        self.assertTrue(self.page.locator('[data-role=map-sites]').is_hidden())
        self.assertIn('Chromosome 1',self.page.locator('[data-role=map-description]').inner_text())
        self.page.locator('[data-role=context]').select_option('all');self.settle()
        self.assertTrue(self.page.locator('[data-role=map-section]').is_hidden())

    def test_utr_and_regulatory_tracks_keep_geometry_hover_and_selection_on_mobile(self):
        data=project_records([{'antisense_5to3':'ACGT'*4+'AC'+'ACGT'[i%4]+'ACGT'[i//4],
            'binding_intervals_bed0_half_open':[['chr1',100220+i*2,100240+i*2]]} for i in range(12)])
        context=data['contexts'][0]
        context.update(reference_chrom='chr1',reference_strand='+',transcript='TX.1',
            exons=[{'start':100200,'end':100500,'index':3,'label':'third annotated exon'}],
            reference_features={'status':'resolved','note':'Saved reference annotation.',
                'segments':[{'chrom':'chr1','start':100200,'end':100500,'exon_index':3,'kind':'utr3','label':'3′ UTR'}]},
            regulatory_sites=[
                {'kind':'rbp','label':'PUM1','chrom':'chr1','start':100225,'end':100240,'strand':'+',
                 'evidence':'eCLIP peak','source':'K562 · ENCODE'},
                {'kind':'mirna','label':'miR-test','chrom':'chr1','start':100245,'end':100252,'strand':'+',
                 'evidence':'Predicted 8mer site','source':'TargetScan'}])
        self.page.evaluate('''()=>{const p=CanvasRenderingContext2D.prototype,fill=p.fillRect;
            window.annotationBars=[];p.fillRect=function(x,y,w,h){
            if(this.canvas.dataset.role==='map'){
              if(x===0&&y===0)window.annotationBars=[];
              if(h===10)window.annotationBars.push({x,y,w,h,color:this.fillStyle});
            }return fill.apply(this,arguments);};}''')
        self.page.set_content(basic_document('Regulatory tracks','',candidate_data=data));self.settle()
        for width in [1440,390]:
            self.page.set_viewport_size({'width':width,'height':1100});self.settle()
            self.assert_clear_labels()
            labels=[x['text'] for x in self.page.evaluate('window.mapDrawing.labels')]
            self.assertTrue(any('3′ UTR' in s for s in labels))
            self.assertTrue(any('RBP · PUM1' in s for s in labels))
            self.assertTrue(any('miRNA · miR-test' in s for s in labels))
            self.assertIn('3′ UTR',self.page.locator('[data-role=map-feature-note]').inner_text())
            bars=self.page.evaluate('window.annotationBars')
            self.assertEqual(len(bars),2)
            self.assertAlmostEqual(bars[0]['w']/bars[1]['w'],2,places=5)  # 16 bp versus 8 bp.
            canvas=self.page.locator('[data-role=map]')
            canvas.hover(position={'x':bars[0]['x']+bars[0]['w']/2,'y':bars[0]['y']+5})
            self.assertIn('K562',self.page.locator('[data-role=map-tip]').inner_text())
            self.assertIn('100,225–100,240',self.page.locator('[data-role=map-tip]').inner_text())
            self.assertFalse(self.page.evaluate('document.documentElement.scrollWidth>innerWidth+1'))
        bar=self.page.evaluate('window.mapDrawing.bars[0]')
        canvas.click(position={'x':bar['x']+bar['w']/2,'y':bar['y']+3});self.settle()
        self.assertIn('1 ASO',self.page.locator('[data-role=basket-count]').inner_text())
        self.page.locator('[data-role=search]').fill('no matching ASO');self.settle()
        self.assertEqual(len(self.page.evaluate('window.annotationBars')),2)
        self.assert_clear_labels()
