"""An offline candidate table, reference map and selection basket."""
import html
import json
from .report_attribute_ui import ATTRIBUTE_HTML, ATTRIBUTE_SCRIPT
from .report_binding_map import BINDING_MAP_SCRIPT
from .report_approaches import approach_html, context_approaches, ordered_contexts
from .report_exon_names import exon_naming_html
from .report_recommendations import recommendations_html
from .report_binding_annotations import annotation_html
from .report_selection_exports import SELECTION_EXPORT_SCRIPT


def pack_data(data):
    """Column dictionaries remove repeated metadata without rounding evidence."""
    rows = [{**r, 'chemistry':{k:v for k,v in r['chemistry'].items() if k!='notation'},
             'notation':r['chemistry']['notation']} for r in data['records']]
    fields = list(rows[0])
    columns, encoded = [], [[] for _ in rows]
    for field in fields:
        values = [r[field] for r in rows]
        unique, lookup, indices = [], {}, []
        for value in values:
            key = json.dumps(value, ensure_ascii=False, separators=(',',':'))
            if key not in lookup:
                lookup[key] = len(unique)
                unique.append(value)
            indices.append(lookup[key])
        use_dictionary = len(unique)<len(values)/2
        columns.append({'name':field, **({'values':unique} if use_dictionary else {})})
        for row,value in zip(encoded,indices if use_dictionary else values):
            row.append(value)
    from .report_attributes import is_color_identity, pack_attributes
    attributes = []
    for attribute in data.get('attributes') or []:
        # Older saved catalogs carry display labels and the same exact values.
        if attribute.get('group') == 'Saved attributes' and is_color_identity(
                attribute['label'].split(' · '), attribute.get('labels') or {}):
            attribute = {**attribute, 'color_identity': True}
        attributes.append(attribute)
    return {**data, 'attributes':pack_attributes(attributes),
            'encoding':'columns-v1', 'record_columns':columns, 'records':encoded}


def explorer_html(data, *, filter_history=None):
    from .report_filter_history_ui import filter_history_html
    history = filter_history_html(filter_history)
    if not data or not data.get('records'):
        return history
    data = {**data, 'contexts': ordered_contexts(data)}
    esc = lambda value: html.escape(str(value if value is not None else ''), quote=True)
    panel_label = ('Recommended ASOs' if (data.get('recommendation_review') or {}).get('status') == 'completed'
                   else 'Design panel' if data.get('kind') == 'design' else 'Saved panel')
    anchor = 'report-' + esc(data['id'])
    facts = ''.join('<div class="report-fact"><span class="report-fact-label">'+esc(f['label'])
        +'</span><span class="report-fact-value">'+esc(f['value'])+'</span><span class="report-fact-note">'
        +esc(f.get('note'))+'</span></div>' for f in data.get('facts') or [])
    approaches = context_approaches(data)
    def option_label(context):
        choices = approaches[context['id']]
        rank = choices[0]['rank'] if len(choices) == 1 else None
        return (f'Approach rank {rank} · ' if rank is not None else '') + context['label']
    options = ''.join('<option value="'+esc(c['id'])+'">'+esc(option_label(c))+'</option>' for c in data['contexts'])
    if len(data['contexts'])>1:
        options += '<option value="all">All design groups</option>'
    payload = json.dumps(pack_data(data), ensure_ascii=False, separators=(',',':'), allow_nan=False).replace('<','\\u003c').replace('&','\\u0026')
    # The initial rows make the actual results readable before scripts load.
    rows = []
    initial_context = data['contexts'][0]['id']
    initial_rows = [r for r in data['records'] if r['context']==initial_context]
    initial_activity = next((a for r in initial_rows for a in r.get('oligoai') or []),None)
    activity_id = initial_activity['scenario_id'] if initial_activity else None
    initial_conditions = initial_activity.get('conditions_id') if initial_activity else None
    combine_contexts = initial_conditions and len({a['scenario_id'] for r in initial_rows for a in r.get('oligoai') or []
                                                  if a.get('conditions_id')==initial_conditions}) > 1
    activity_note = ''
    if initial_activity:
        label = ('All RNA contexts · ' + initial_activity['conditions_label']) if combine_contexts else initial_activity['label']
        activity_note = 'OligoAI predicts RNA reduction in vitro. ' + label + '.'
        assumed = [label for field,label in (('dose_basis','dose'),('method_basis','delivery'))
                   if initial_activity.get(field) == 'assumed_reference_scenario']
        if assumed and not initial_activity.get('unresolved_conditions'):
            activity_note += ' Reference ' + ' and '.join(assumed) + '.'
        if initial_activity.get('unmodeled_conditions'):
            activity_note += ' Experiment details outside the model’s inputs: ' + '; '.join(initial_activity['unmodeled_conditions']) + '.'
    def activity_values(row):
        return [a['value'] for a in row.get('oligoai') or [] if a['value'] is not None and
                (a.get('conditions_id')==initial_conditions if combine_contexts else a['scenario_id']==activity_id)]
    if any(activity_values(r) for r in initial_rows) and not any(r['effect'] is not None for r in initial_rows):
        initial_rows.sort(key=lambda r:(not bool(activity_values(r)),-min(activity_values(r) or [0])))
    for row in initial_rows[:25]:
        activity = activity_values(row)
        activity_text = ((f'{min(activity):.1f}%' + (f'–{max(activity):.1f}%' if min(activity)!=max(activity) else ''))
                         if activity else 'No saved prediction')
        values = ['',row['aso'],row['location'],
                  f"{row['effect']:+.3f}" if row['effect'] is not None else 'Unavailable',
                  f"{row['gc']*100:.0f}%" if row['gc'] is not None else 'Unavailable',
                  ' / '.join(str(int(v)) if v is not None else '—' for v in row['off_targets']),
                  activity_text,row['guard']['summary']]
        columns = ('select', 'aso', 'start', 'effect', 'gc', 'off_targets', 'oligoai', 'details')
        rows.append('<tr>'+''.join('<td data-column="'+column+'">'+esc(v)+'</td>'
                                   for column, v in zip(columns, values))+'</tr>')
    has_activity = any(r.get('oligoai') or data.get('kind')!='chemistry' and r.get('oligoai_applicable') for r in data['records'])
    classes = 'report-explorer' + ('' if has_activity else ' report-no-oligoai')
    filters = '''<div class="report-controls report-basic-filters" data-role="basic-filters">
<label>Sequence, name or position<input type="search" name="aso-search" data-role="search" placeholder="Search ASOs…" autocomplete="off" spellcheck="false"></label>
<label data-role="effect-control">Minimum prediction<input type="number" name="minimum-prediction" step="any" data-role="minimum" placeholder="Any"></label>
<label data-role="activity-minimum-control" hidden>Minimum RNA reduction (%)<input type="number" name="minimum-rna-reduction" step="any" data-role="activity-minimum" placeholder="Any"></label>
<label class="report-check"><input type="checkbox" data-role="panel-only">'''+panel_label+''' only</label>
<label class="report-check"><input type="checkbox" data-role="basket-only">In my basket</label>
<label class="report-check"><input type="checkbox" data-role="clean-only">Zero exact or 1-mismatch off-targets</label>
<button type="button" class="report-button" data-action="reset">Reset all filters</button></div>'''
    navigation = ('<nav class="report-section-nav" data-role="section-nav" aria-label="Report sections">'
        + ('<a class="report-jump" href="#'+anchor+'-recommendations">'
           + ('Recommendations' if (data.get('recommendation_review') or {}).get('status') == 'completed' else 'Panel review')
           + '</a>' if data.get('recommendation_review') else '')
        + '<a class="report-jump" data-role="map-jump" href="#'+anchor+'-binding" hidden>Binding map</a>'
        + '<a class="report-jump" href="#'+anchor+'-asos">ASO table</a>'
        + '<a class="report-jump" href="#'+anchor+'-basket">Your basket</a>'
        + ('<a class="report-jump" href="#filtered-candidates">Filtered candidates (' + str(len(filter_history['records'])) + ')</a>' if history else '') + '</nav>')
    return '<section class="'+classes+'''" id="candidate-explorer" aria-label="Explore ASO candidates">
'''+navigation+'''
<div class="report-facts">'''+facts+'''</div>
<div class="report-goal"><span class="report-goal-label">What this run was asked to do</span><p data-role="objective">'''+esc(data['objective'])+'''</p></div>
'''+recommendations_html(data)+'''
'''+('<h2>Targets and approaches</h2>' if data.get('kind') == 'design' else '')+'''
<div class="report-controls"><label class="report-context">Target and approach<select data-role="context" aria-label="Target and approach">'''+options+'''</select></label>
<p class="report-sub">Choose a target to explore its predictions and ASOs.</p></div>
'''+approach_html(data)+exon_naming_html(data['contexts'])+'''
<div class="report-controls" data-role="activity-controls" hidden><label class="report-context">OligoAI RNA context and conditions<select data-role="activity-context" aria-label="OligoAI RNA context and conditions"></select></label></div>
<p class="report-sub" data-role="activity-note"'''+('' if activity_note else ' hidden')+'''>'''+esc(activity_note)+'''</p>
<div class="report-exploration-layout" data-role="exploration-layout">
'''+ATTRIBUTE_HTML.replace('<!-- BASIC_FILTERS -->', filters)+'''
<section class="report-binding-view" data-role="map-section" id="'''+anchor+'''-binding" tabindex="-1" hidden><h2>Where the ASOs bind</h2><label class="report-sub" data-role="chromosome-control" hidden>Chromosome <select data-role="map-chromosome" aria-label="Map chromosome"></select></label><p class="report-map-description" data-role="map-description"></p>
'''+annotation_html(data['contexts'])+'''
<p class="report-sub" data-role="map-feature-note"></p>
<p class="report-sub" data-role="map-target-note" hidden></p>
<div class="report-map-controls"><label data-role="map-region-control" hidden>Map region <select data-role="map-region" aria-label="Map region"></select></label><label data-role="map-values-control" hidden><input type="checkbox" data-role="map-values"> Show values by position</label></div>
<div class="report-map" tabindex="0" role="region" aria-label="Scrollable map of ASO binding sites"><canvas data-role="map" role="img" aria-label="ASO binding map; use the candidate table to select ASOs with a keyboard"></canvas></div>
<div class="report-legend"><span class="report-gradient"></span><span data-role="legend"></span><div class="report-category-legend" data-role="category-legend"></div></div>
<div class="report-pagination"><span data-role="map-status" aria-live="polite"></span><div data-role="map-paging"><button class="report-button" data-action="map-previous">Previous rows</button><button class="report-button" data-action="map-next">Next rows</button></div></div>
<p class="report-map-tip" data-role="map-tip" aria-live="polite">Select an ASO in the map or table to add it to your basket.</p>
<details class="report-map-sites" data-role="map-sites" hidden><summary data-role="map-sites-summary">Splice sites</summary><p class="report-sub">Splice sites are RNA cut-and-join positions. Teal marks the target for this approach.</p><div class="report-map-site-list" tabindex="0" role="region" aria-label="Recorded splice sites"><table data-report-native="true"><thead><tr><th scope="col">Splice site</th><th scope="col">Map region</th></tr></thead><tbody data-role="map-sites-body"></tbody></table></div></details></section>
</div>
<h2 id="'''+anchor+'''-asos" tabindex="-1">Explore all ASOs</h2>
<p class="report-sub" data-role="metric-note"></p>
<p class="report-sub"><span data-role="count" aria-live="polite">'''+str(len(data['records']))+''' results available.</span> <span data-role="screen-description">Off-targets are other sequences an ASO may also bind. Counts below show exact / 1-mismatch / 2-mismatch matches.</span></p>
<div class="report-table-wrap" tabindex="0" role="region" aria-label="ASO candidate table"><table class="report-table" data-role="candidate-table" data-report-native="true"><thead><tr>
<th scope="col">Select</th><th scope="col" data-sort="aso"><button type="button">ASO · 5′→3′</button></th>
<th scope="col" data-sort="start"><button type="button">Binding location</button></th><th scope="col" data-sort="effect"><button type="button">Predicted splice effect</button></th>
<th scope="col" data-sort="gc"><button type="button" title="Percentage of G and C bases">GC</button></th><th scope="col" data-sort="off_targets"><button type="button">Off-targets</button></th><th scope="col" data-sort="oligoai"><button type="button" title="Predicted RNA reduction in vitro. Sorting uses the lowest prediction when an ASO has several binding sites in this RNA context.">OligoAI RNA reduction</button></th><th scope="col">Checks &amp; details</th></tr></thead>
<tbody>'''+''.join(rows)+'''</tbody></table></div>
<div class="report-pagination"><label>Rows per page <select data-role="page-size" aria-label="Rows per page"><option>25</option><option>50</option><option>100</option></select></label>
<span data-role="page-status" aria-live="polite"></span><div><button class="report-button" data-action="previous">Previous</button><button class="report-button" data-action="next">Next</button></div></div>
<div class="report-basket" role="region" aria-label="Selected ASO basket" data-role="basket" data-docked="false" id="'''+anchor+'''-basket" tabindex="-1"><div class="report-basket-main"><span class="report-basket-count" data-role="basket-count" aria-live="polite">No ASOs selected</span>
<div class="report-basket-actions"><button class="report-button" data-action="add-panel">Add '''+panel_label[0].lower()+panel_label[1:]+'''</button><button class="report-button" data-action="add-top" disabled>Add matching ASOs</button>
<button class="report-button report-button-primary" data-action="csv" disabled>Download CSV</button><button class="report-button" data-action="fasta" disabled>FASTA</button><button class="report-button" data-action="clear" disabled>Clear basket</button><button class="report-button" data-action="undo-clear" hidden>Undo clear</button></div></div>
<details><summary>Selected ASOs and ordering options</summary><ul class="report-basket-items" data-role="basket-items"></ul>
<div class="report-order-fields"><label>Number to add<input type="number" data-role="top-count" min="1" max="10000" value="10" aria-describedby="'''+anchor+'''-count-error"></label><label>Scale<input data-role="scale" placeholder="Specify with supplier"></label><label>Purification<input data-role="purification" placeholder="Specify with supplier"></label></div>
<p class="report-filter-error" data-role="top-count-error" id="'''+anchor+'''-count-error" aria-live="polite"></p>
<p class="report-sub">CSV has one row per selected ASO, with each context’s predictions, conditions, bindings and safeguards. Full context evidence includes every saved attribute and reference definition. The order sheet uses each ASO’s saved ordering notation exactly.</p>
<p class="report-notice" data-role="order-notice" hidden></p><p class="report-sub" data-role="storage-notice" hidden></p>
<button class="report-button" data-action="order" disabled>Download order sheet</button><button class="report-button" data-action="evidence" disabled>Full context evidence (JSON)</button></details></div>
<noscript><p class="report-explorer-noscript">The first 25 records are shown here. Enable JavaScript for search, the map, selection and exports. Saved analysis and file links remain below.</p></noscript>
<script type="application/json" class="report-explorer-data">'''+payload+'''</script></section>''' + history


EXPLORER_SCRIPT = r'''<script>
(()=>{'use strict';
const esc=value=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const fmt=value=>finite(value)?(value>0?'+':'')+value.toFixed(3):'Unavailable';
const percent=value=>finite(value)?value.toFixed(1)+'%':'Unavailable';
const ramp=value=>{if(!finite(value)||value<=0)return '#e9eded';const a=Math.pow(Math.min(value,1),.7);return `rgb(${Math.round(226-190*a)},${Math.round(234-106*a)},${Math.round(234-106*a)})`;};
const csvCell=value=>{let text=String(value??'');if(typeof value==='string'&&!['+','-'].includes(text)&&'=+@-'.includes(text.trimStart().charAt(0))&&text.trim())text="'"+text;return '"'+text.replaceAll('"','""')+'"';};
const download=(name,text,type)=>{const url=URL.createObjectURL(new Blob([text],{type:type||'text/plain;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();setTimeout(()=>{link.remove();URL.revokeObjectURL(url);},1000);};
/* ATTRIBUTE_HELPERS */
/* BINDING_MAP_HELPERS */
/* SELECTION_EXPORT_HELPERS */
function setup(root){
 if(root.dataset.initialized)return;root.dataset.initialized='true';
 const packet=root.querySelector('.report-explorer-data');if(!packet)return;
 let data;try{data=JSON.parse(packet.textContent);}catch{root.insertAdjacentHTML('afterbegin','<p class="report-notice">Interactive tools could not load. The saved table and analysis remain available below.</p>');return;}
 if(data.encoding==='columns-v1')data.records=data.records.map(values=>{
  const row={};data.record_columns.forEach((column,i)=>row[column.name]=column.values?column.values[values[i]]:values[i]);
  row.chemistry={...row.chemistry,notation:row.notation};delete row.notation;return row;
 });
 const q=role=>root.querySelector(`[data-role="${role}"]`),action=name=>root.querySelector(`[data-action="${name}"]`);
 const evidence=document.getElementById('reader-analysis');
 if(evidence){evidence.tabIndex=-1;const link=document.createElement('a');link.className='report-jump';link.href='#reader-analysis';link.textContent='Evidence';q('section-nav').append(link);}
 const rows=data.records,byId=new Map(rows.map(r=>[r.id,r])),contexts=new Map(data.contexts.map(c=>[c.id,c]));
 const byMolecule=new Map();rows.forEach(r=>{if(!byMolecule.has(r.molecule))byMolecule.set(r.molecule,[]);byMolecule.get(r.molecule).push(r);});
 const basket=new Map(),storageKey='modal-aso-selection:'+data.id;
 const panelRows=rows.filter(r=>r.panel_rank!==null).sort((a,b)=>a.panel_rank-b.panel_rank);
 const panelLabel=data.recommendation_review?.status==='completed'?'Recommended ASOs':data.kind==='design'?'Design panel':'Saved panel';
 let clearedBasket=null;
 const activityViews=new Map();let activityScenarios=[];
 let page=0,mapPage=0,pageSize=25,sort='effect',ascending=false,filtered=[],mapHits=[],mapSiteHits=[],mapAnnotationHits=[],mapScope=null,mapSitesHTML='',mapFrame=0,printing=false;
 try{const saved=JSON.parse(localStorage.getItem(storageKey)||'[]');if(Array.isArray(saved))saved.forEach(id=>{const r=byId.get(id);if(r)basket.set(r.molecule,r);});}catch{q('storage-notice').textContent='Selections stay available while this report is open. Browser storage is unavailable.';q('storage-notice').hidden=false;}
 let scopeId=null,scopedRows=[],baseKey=null,baseRows=[];
 const scopeRows=()=>{const id=q('context').value;if(id!==scopeId){scopeId=id;scopedRows=id==='all'?rows:rows.filter(r=>r.context===id);}return scopedRows;};
 const activityFor=row=>activityViews.get(row.id)||{entries:[],values:[],value:null,missing:0};
 const attributeView=setupAttributes({root,rows,attributes:data.attributes,scopeRows,baseRows:baseVisible,activityFor,
  changed:()=>{page=0;mapPage=0;renderTable();}});
 function configureActivity(){
  const entries=scopeRows().flatMap(r=>r.oligoai||[]),withReference=entries.filter(a=>a.transcript_id||a.context_basis);
  const contexts=[...new Map((withReference.length?withReference:entries).map(a=>[a.scenario_id,a])).values()];
  const conditions=[...new Map(contexts.filter(a=>a.conditions_id).map(a=>[a.conditions_id,a])).values()];
  const combined=conditions.filter(a=>contexts.filter(b=>b.conditions_id===a.conditions_id).length>1)
   .map(a=>({...a,scenario_id:'conditions:'+a.conditions_id,combined:true,label:'All RNA contexts · '+a.conditions_label}));
  activityScenarios=[...combined,...contexts];
  const selected=q('activity-context').value;
  q('activity-context').innerHTML=activityScenarios.map((a,i)=>`<option value="${esc(a.scenario_id)}">${esc(a.label)}${activityScenarios.some(b=>b!==a&&b.label===a.label)?' · record '+(i+1):''}</option>`).join('');
  if(activityScenarios.some(a=>a.scenario_id===selected))q('activity-context').value=selected;
  q('activity-controls').hidden=activityScenarios.length<2;
  updateActivity();
 }
 function updateActivity(){
  const scenario=activityScenarios.find(a=>a.scenario_id===q('activity-context').value);
  activityViews.clear();
  scopeRows().forEach(r=>{
   const all=r.oligoai||[];let entries=all.filter(a=>scenario?.combined?a.conditions_id===scenario.conditions_id:a.scenario_id===scenario?.scenario_id);
   if(!entries.length)entries=all.filter(a=>!a.transcript_id&&!a.context_basis&&['dosage_nm','transfection_method'].every(k=>a[k]==null||a[k]===scenario?.[k]));
   const values=entries.map(a=>a.value).filter(finite);
   activityViews.set(r.id,{entries,values,value:values.reduce((low,v)=>low===null?v:Math.min(low,v),null),missing:entries.length-values.length});
  });
  const applicable=scopeRows().filter(r=>(data.kind!=='chemistry'&&r.oligoai_applicable)||(r.oligoai||[]).length),hasActivity=applicable.length>0;
  const predicted=new Set(applicable.filter(r=>finite(activityFor(r).value)).map(r=>r.molecule)).size;
  const total=new Set(applicable.map(r=>r.molecule)).size;
  root.classList.toggle('report-no-oligoai',!hasActivity);q('activity-note').hidden=!hasActivity;
  q('activity-minimum-control').hidden=!predicted;
  const assumed=scenario&&!scenario.unresolved_conditions?.length?['dose_basis','method_basis'].filter(k=>scenario[k]==='assumed_reference_scenario'):[];
  const basis=assumed.length===2?' Reference dose and delivery.':assumed.length?' Reference '+(assumed[0]==='dose_basis'?'dose.':'delivery.'):'';
  const ranges=applicable.some(r=>activityFor(r).values.length>1)?' Ranges cover separate RNA contexts or binding sites; sorting uses the lower value.':'';
  const unmodeled=scenario?.unmodeled_conditions?.length?' Experiment details outside the model’s inputs: '+scenario.unmodeled_conditions.join('; ')+'.':'';
  q('activity-note').textContent='OligoAI predicts RNA reduction in vitro. '+predicted.toLocaleString()+' of '+total.toLocaleString()+' ASOs have predictions for these conditions.'+(scenario?' '+scenario.label+'.':'')+basis+ranges+unmodeled;
 }
 function activityCell(row){
  const a=activityFor(row);if(!a.entries.length)return '<span class="report-sub">No saved prediction</span>';
  const labels={disabled:'Disabled',unsupported:'Unsupported chemistry',error:'Prediction failed',pending:'Pending',unavailable:'Unavailable',predicted:'Predicted'};
  let summary=labels[a.entries[0].status]||'Unavailable';
  if(a.values.length){const low=a.value,high=a.values.reduce((high,v)=>Math.max(high,v),low);
   summary=`<span class="report-effect" style="background:${ramp(low/100)};color:${low>55?'#fff':'#1b1f23'}">${percent(low)}${high!==low?'–'+percent(high):''}</span>`;
   if(a.missing)summary+=` <span class="report-sub">${a.missing} unavailable</span>`;
  }
  const details=a.entries.map(v=>`<p><strong>${finite(v.value)?percent(v.value)+' predicted RNA reduction':esc(labels[v.status]||'Unavailable')}</strong><br>${esc(v.label)}${v.location?'<br>'+esc(v.location):''}${v.reason?'<br>'+esc(v.reason):''}${v.unmodeled_conditions?.length?'<br>Experiment details outside the model’s inputs: '+esc(v.unmodeled_conditions.join('; ')):''}${v.masked_flank_nt?'<br>'+esc(v.masked_flank_nt)+' flanking bases were unavailable.':''}</p>`).join('');
  return `<details class="report-activity-details"><summary>${summary}</summary><div class="report-check-text">${details}</div></details>`;
 }
 function baseVisible(){
  const query=q('search').value.trim().toUpperCase(),minimum=q('minimum').value.trim(),activityMinimum=q('activity-minimum').value.trim();
  const panel=q('panel-only').checked,selected=q('basket-only').checked,clean=q('clean-only').checked;
  const key=JSON.stringify([q('context').value,q('activity-context').value,query,minimum,activityMinimum,panel,selected,clean]);
  if(key===baseKey)return baseRows;
  baseKey=key;baseRows=scopeRows().filter(r=>(!panel||r.panel_rank!==null)&&(!selected||basket.has(r.molecule))&&
   (!clean||(r.off_target_complete&&r.off_targets[0]===0&&r.off_targets[1]===0))&&
   (!minimum||(finite(r.effect)&&(contexts.get(q('context').value)?.direction==='decrease'?r.effect<=Number(minimum):r.effect>=Number(minimum))))&&
   (!activityMinimum||(finite(activityFor(r).value)&&!activityFor(r).missing&&activityFor(r).value>=Number(activityMinimum)))&&
   (!query||[r.aso,r.name,r.location,contexts.get(r.context)?.label].some(v=>String(v||'').toUpperCase().includes(query))));
  return baseRows;
 }
 function visible(){
  let result=baseVisible().filter(r=>attributeView.matches(r));
  const multi=q('context').value==='all';
  return result.sort((a,b)=>{
   if(multi&&sort!=='panel_rank'&&a.context!==b.context)return data.contexts.findIndex(c=>c.id===a.context)-data.contexts.findIndex(c=>c.id===b.context);
   const sortValue=r=>sort.startsWith('attribute:')?attributeView.sortValue(sort.slice(10),r):sort==='oligoai'?activityFor(r).value:sort==='off_targets'?r.off_targets[0]:r[sort];
   let x=sortValue(a),y=sortValue(b);
   if(x===null||x===undefined)return y===null||y===undefined?a.order-b.order:1;if(y===null||y===undefined)return -1;
   const delta=typeof x==='string'?x.localeCompare(y):x-y;return (ascending?delta:-delta)||a.order-b.order;
  });
 }
 function renderTable(){
  const tableAttributes=attributeView.tableAttributes(),header=q('candidate-table').querySelector('thead tr');
  const signature=tableAttributes.map(a=>a.id).join(',');
  if(header.dataset.attributeColumns!==signature){
   header.querySelectorAll('[data-attribute-column]').forEach(th=>th.remove());
   header.lastElementChild.insertAdjacentHTML('beforebegin',tableAttributes.map(a=>`<th scope="col" class="report-attribute-column" data-attribute-column="${a.id}" data-sort="attribute:${a.id}"><button type="button" title="${esc(a.description)}">${esc(a.label)}</button></th>`).join(''));
   header.dataset.attributeColumns=signature;
   if(sort.startsWith('attribute:')&&!tableAttributes.some(a=>'attribute:'+a.id===sort)){sort='order';ascending=true;}
  }
  filtered=visible();const total=filtered.length,size=printing?Math.max(1,total):pageSize;
  page=Math.max(0,Math.min(page,Math.max(0,Math.ceil(total/size)-1)));
  const shown=filtered.slice(page*size,(page+1)*size),scoped=scopeRows();
  q('candidate-table').querySelector('tbody').innerHTML=shown.map(r=>{
   const color=attributeView.effectColor(r),white=finite(r.effect)&&Math.abs(r.effect)>.55;
   const raw=r.predictions.length?`<dt>Deletion / replacement prediction</dt><dd>${r.predictions.map(fmt).join(' / ')}</dd>`:'';
   const checks=r.guard.details.map(g=>`<p>${esc(g.text)}${g.reason?'<br>'+esc(g.reason):''}</p>`).join('');
   const chemistry=esc(r.chemistry.detail||r.chemistry.label);
   const recommendation=r.recommendation?.why_chosen?`<dt>Panel role</dt><dd>${esc(r.recommendation.panel_role)}</dd><dt>Tradeoff</dt><dd>${esc(r.recommendation.tradeoff)}</dd>`:'';
   const targetSet=r.target_context?`<dt>Target selectivity and coverage</dt><dd>${esc(r.target_context.text)}</dd>`:'';
   const screen=r.off_targets.map(v=>finite(v)?v.toLocaleString():'—').join(' / ');
   return `<tr data-record="${esc(r.id)}" data-panel="${r.panel_rank!==null}" aria-selected="${basket.has(r.molecule)}"><td><input type="checkbox" data-pick="${esc(r.id)}" aria-label="Select ${esc(r.name)}" ${basket.has(r.molecule)?'checked':''}></td>
   <td><span class="report-attribute-swatch" style="background:${attributeView.color(r)}" title="${esc(attributeView.describe(r))}" aria-hidden="true"></span><span class="report-sequence">${esc(r.aso)}</span>${r.panel_rank!==null?` <span class="report-badge report-badge-panel">${data.recommendation_review?.status==='completed'?'Recommendation':'Panel'} ${r.panel_rank}</span>`:''}<div class="report-sub">${esc(r.chemistry.label)}</div></td>
   <td data-column="start"><span class="report-sub">${esc(r.location)}</span>${q('context').value==='all'?'<div class="report-sub">'+esc(contexts.get(r.context)?.label)+'</div>':''}</td>
   <td class="report-number" data-column="effect"><span class="report-effect" style="background:${color};color:${white?'#fff':'#1b1f23'}">${fmt(r.effect)}</span></td><td class="report-number" data-column="gc">${finite(r.gc)?(r.gc*100).toFixed(0)+'%':'—'}</td>
   <td data-column="off_targets"><span class="report-badge">${screen}</span><div class="report-sub">${esc(r.off_target_unit)}${r.off_target_complete?'':r.off_target_status==='unavailable'?' · search result missing':r.off_target_status==='failed'?' · search failed':' · search incomplete'}</div></td>
   <td data-column="oligoai" data-activity="${esc(r.id)}">${activityCell(r)}</td>
   ${tableAttributes.map(a=>`<td class="report-attribute-column">${esc(attributeView.cell(a,r))}</td>`).join('')}
   <td><details><summary>${['passed','review','incomplete'].includes(r.guard.status)?esc(r.guard.summary):'View ASO details'}</summary><div class="report-check-text">${checks||esc(r.guard.summary)}<dl><dt>Selection</dt><dd>${esc(r.selection_note)}</dd>${recommendation}<dt>Chemistry</dt><dd>${chemistry}</dd>${raw}${targetSet}<dt>Target and approach</dt><dd>${esc(contexts.get(r.context)?.label)}</dd><dt>Saved rank</dt><dd>${r.native_rank??'Unavailable'}</dd><dt>Ordering notation</dt><dd><code>${esc(r.chemistry.notation||'Unavailable')}</code></dd></dl></div></details></td></tr>`;
  }).join('')||`<tr><td colspan="${8+tableAttributes.length}" class="report-empty"><p>No ASOs match these filters.</p><button type="button" class="report-button" data-action="empty-reset">Clear filters</button></td></tr>`;
  const molecules=new Set(filtered.map(r=>r.molecule)).size;
  q('count').textContent=`${total.toLocaleString()} of ${scoped.length.toLocaleString()} results · ${molecules.toLocaleString()} ASOs.`;
  q('page-status').textContent=total?`${(page*size+1).toLocaleString()}–${Math.min((page+1)*size,total).toLocaleString()} of ${total.toLocaleString()}`:'No matching records';
  action('previous').disabled=page===0;action('next').disabled=(page+1)*size>=total;
  root.querySelectorAll('th[data-sort]').forEach(th=>th.setAttribute('aria-sort',th.dataset.sort===sort?(ascending?'ascending':'descending'):'none'));
  attributeView.render();
  updateBulkAction();
  scheduleMap();
 }
 function nextMatchingRows(){
  const amount=Number(q('top-count').value);
  if(!Number.isInteger(amount)||amount<1||amount>10000||!attributeView.valid())return [];
  const seen=new Set(basket.keys()),chosen=[];
  for(const row of filtered){if(seen.has(row.molecule))continue;seen.add(row.molecule);chosen.push(row);if(chosen.length===amount)break;}
  return chosen;
 }
 function updateBulkAction(){
  const amount=Number(q('top-count').value),validAmount=Number.isInteger(amount)&&amount>=1&&amount<=10000;
  q('top-count').setAttribute('aria-invalid',String(!validAmount));
  q('top-count-error').textContent=validAmount?'':'Choose a whole number from 1 to 10,000.';
  const count=nextMatchingRows().length;
  action('add-top').disabled=!count;
  action('add-top').textContent=count?`Add ${count.toLocaleString()} matching ASO${count===1?'':'s'}`:'Add matching ASOs';
 }
 function save(){try{localStorage.setItem(storageKey,JSON.stringify([...basket.values()].map(r=>r.id)));}catch{q('storage-notice').hidden=false;q('storage-notice').textContent='Selections stay available while this report is open. Browser storage is unavailable.';}}
 function renderBasket(){
  baseKey=null;
  const n=basket.size;q('basket-count').textContent=n?`${n.toLocaleString()} ASO${n===1?'':'s'} selected`:'No ASOs selected';
  q('basket').dataset.docked=String(!!n||!!clearedBasket);
  updateLayoutSpace();
  ['csv','fasta','clear','evidence'].forEach(a=>action(a).disabled=!n);
  action('undo-clear').hidden=!clearedBasket;
  root.querySelectorAll('[data-recommendation-pick]').forEach(button=>{
   const row=byId.get(button.dataset.recommendationPick);if(!row)return;
   const selected=basket.has(row.molecule);button.dataset.selected=String(selected);
   button.textContent=selected?`Remove ASO ${row.panel_rank} from basket`:`Add ASO ${row.panel_rank} to basket`;
  });
  const panelAdded=panelRows.length>0&&panelRows.every(r=>basket.has(r.molecule));
  root.querySelectorAll('[data-action="add-panel"],[data-action="add-recommendations"]').forEach(button=>{
   button.hidden=!panelRows.length;button.disabled=panelAdded;
   button.textContent=panelAdded?panelLabel+' added':'Add '+panelLabel[0].toLowerCase()+panelLabel.slice(1);
  });
  const selected=[...basket.values()],missing=selected.filter(r=>!r.chemistry.notation).length;
  action('order').disabled=!n||missing>0;
  q('basket-items').innerHTML=selected.slice(0,100).map(r=>`<li><span><b>${esc(r.name)}</b> · ${esc(r.aso)} · ${esc(r.chemistry.label)}</span><button class="report-button" data-remove="${esc(r.molecule)}" aria-label="Remove ${esc(r.name)}">Remove</button></li>`).join('')+(n>100?`<li>${n-100} more selections are included in exports.</li>`:'');
  const issues=[...new Set(selected.flatMap(r=>r.chemistry.issues||[]))];
  q('order-notice').hidden=!issues.length;q('order-notice').textContent=(missing?`${missing} selections need ordering notation. `:'')+issues.join(' ');
 }
 function toggle(id){
  const row=byId.get(id);if(!row)return;
  const focused=document.activeElement?.dataset.pick;
  clearedBasket=null;
  if(basket.has(row.molecule))basket.delete(row.molecule);else basket.set(row.molecule,row);
  save();renderBasket();renderTable();
  if(focused){const next=[...root.querySelectorAll('[data-pick]')].find(el=>el.dataset.pick===focused);
   (next||q('candidate-table').closest('.report-table-wrap')).focus({preventScroll:true});}
 }
 function reset(){q('search').value='';q('minimum').value='';q('activity-minimum').value='';q('panel-only').checked=false;q('basket-only').checked=false;q('clean-only').checked=false;attributeView.clear();page=0;mapPage=0;renderTable();}
 function contextChanged(){
  const context=contexts.get(q('context').value),hasEffects=scopeRows().some(r=>finite(r.effect));
  root.querySelectorAll('[data-approach-context]').forEach(el=>{el.hidden=el.dataset.approachContext!==q('context').value;});
  root.querySelectorAll('[data-exon-context]').forEach(el=>{el.hidden=el.dataset.exonContext!==q('context').value;});
  root.querySelectorAll('[data-binding-context]').forEach(el=>{el.hidden=el.dataset.bindingContext!==q('context').value;});
  q('metric-note').textContent=context?.metric_note||'Each design group keeps its own target, prediction units and recorded ranking.';
  configureActivity();
  attributeView.refresh();
  const hasActivity=scopeRows().some(r=>finite(activityFor(r).value));
  if(hasActivity&&!hasEffects)q('metric-note').textContent='ASOs with saved activity predictions appear first. Each ASO’s details include its original design order.';
  root.classList.toggle('report-no-predictions',!hasEffects);
  const hasScreen=data.kind!=='chemistry'||scopeRows().some(r=>r.off_targets.some(finite)||r.off_target_status!=='unavailable');
  root.classList.toggle('report-no-screening',!hasScreen);q('clean-only').closest('label').hidden=!hasScreen;q('screen-description').hidden=!hasScreen;
  root.classList.toggle('report-no-locations',!scopeRows().some(r=>r.intervals.length));
  root.classList.toggle('report-no-gc',!scopeRows().some(r=>finite(r.gc)));
  const hasPanel=scopeRows().some(r=>r.panel_rank!==null);
  q('panel-only').closest('label').hidden=!hasPanel;if(!hasPanel)q('panel-only').checked=false;
  action('add-panel').hidden=!rows.some(r=>r.panel_rank!==null);
  if(data.contexts.length===1)q('context').closest('.report-controls').hidden=true;
  const effectLabel=(context?.direction==='decrease'?'Maximum':'Minimum')+' predicted splice effect';
  q('effect-control').firstChild.textContent=effectLabel;q('minimum').setAttribute('aria-label',effectLabel);
  q('effect-control').hidden=!hasEffects||!context;q('minimum').value='';q('activity-minimum').value='';sort=hasEffects&&context?'effect':hasActivity?'oligoai':'order';ascending=sort==='order'||sort==='effect'&&context?.direction==='decrease';page=0;mapPage=0;renderTable();
 }
 function scheduleMap(){cancelAnimationFrame(mapFrame);mapFrame=requestAnimationFrame(drawMap);}
 function drawMap(){
  const context=contexts.get(q('context').value),section=q('map-section');
  const scoped=context?scopeRows().filter(r=>r.intervals.length):[];
  const chromKey=value=>String(value).startsWith('chr')?String(value).slice(3):String(value);
  const chromosomes=[...new Set(scoped.flatMap(r=>r.intervals.map(p=>chromKey(p.chrom))))];
  section.hidden=!context||!scoped.length;q('map-jump').hidden=section.hidden;
  mapHits=[];mapSiteHits=[];mapAnnotationHits=[];
  if(section.hidden)return;
  let chromosome=q('map-chromosome').value;if(!chromosomes.includes(chromosome))chromosome=chromosomes[0];
  q('map-chromosome').innerHTML=chromosomes.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join('');q('map-chromosome').value=chromosome;
  q('chromosome-control').hidden=chromosomes.length<2;
  const onChromosome=r=>r.intervals.length&&r.intervals.every(p=>chromKey(p.chrom)===chromosome);
  const annotations=(context.regulatory_sites||[]).filter(s=>chromKey(s.chrom)===chromosome);
  const all=scoped.filter(onChromosome),regions=bindingRegions(all,annotations),sites=(context.sites||[]).filter(s=>chromKey(s.chrom)===chromosome);
  if(!regions.length){section.hidden=true;q('map-jump').hidden=true;return;}
  const regionFor=position=>regions.findIndex(([a,b])=>position>=a&&position<=b);
  const scopeKey=context.id+':'+chromosome;
  if(scopeKey!==mapScope){
   mapScope=scopeKey;mapPage=0;
   q('map-region').innerHTML='<option value="all">All binding regions</option>'+regions.map(([a,b],i)=>`<option value="${i}">Region ${i+1} · ${a.toLocaleString()}–${b.toLocaleString()}</option>`).join('');
   // A large region catalog opens on a target-containing region. The selector
   // keeps every other region, including an overview, one action away.
   const target=sites.find(s=>s.target&&regionFor(s.position)>=0);
   q('map-region').value=regions.length>4?String(target?regionFor(target.position):0):'all';
  }
  q('map-region-control').hidden=regions.length<2;
  const region=q('map-region').value,segments=region==='all'?regions:[regions[Number(region)]];
  const metric=attributeView.metric(),value=r=>metric?.value(r),direction=metric?.direction==='lower'||metric?.id==='effect'&&context?.direction==='decrease'?1:-1;
  const hasMetric=metric?.kind==='number'&&metric.exists,showValues=hasMetric&&q('map-values').checked;
  q('map-values-control').hidden=!hasMetric;
  const tickWidth=showValues?Math.max(...[metric.low,metric.high,(metric.low+metric.high)/2].map(v=>metric.tick(v).length*6+12)):0;
  const canvas=q('map'),padding=getComputedStyle(canvas.parentElement);
  const availableWidth=canvas.parentElement.clientWidth-parseFloat(padding.paddingLeft)-parseFloat(padding.paddingRight);
  const width=Math.max(240,Math.floor(availableWidth),segments.length>4?Math.min(4096,segments.length*180):0),left=Math.min(180,Math.max(16,tickWidth)),right=16;
  const axes=bindingAxes(segments,width,left,right),axisFor=p=>axes.find(a=>p>=a.a&&p<=a.b+1);
  const annotationTracks=bindingAnnotationTracks(annotations,axes),annotationHeight=annotationTracks.length*42;
  const x=p=>{const a=axisFor(p);return a?a.x+(p-a.a)/(a.b-a.a+1)*a.width:null;};
  const candidates=filtered.filter(onChromosome).filter(r=>r.intervals.some(p=>x(p.start)!==null));
  candidates.sort((a,b)=>{const x=value(a),y=value(b);return x==null?(y==null?a.order-b.order:1):y==null?-1:(x-y)*direction||(a.native_rank??a.order)-(b.native_rank??b.order);});
  const lanes=[],packed=[];
  function laneSlot(blocks,lo,hi){
   let left=0,right=blocks.length;
   while(left<right){const middle=(left+right)>>>1;if(blocks[middle][1]+4<lo)left=middle+1;else right=middle;}
   return left<blocks.length&&blocks[left][0]<=hi+4?-1:left;
  }
  candidates.forEach(r=>{const blocks=r.intervals.map(p=>[x(p.start),x(p.end+1)]).filter(p=>p.every(v=>v!==null)).map(([lo,hi])=>[lo,Math.max(hi,lo+2)]);if(!blocks.length)return;
   const lo=Math.min(...blocks.map(b=>b[0])),hi=Math.max(...blocks.map(b=>b[1]));let lane=0,slot=0;
   while(lane<lanes.length&&(slot=laneSlot(lanes[lane],lo,hi))<0)lane++;
   if(!lanes[lane]){lanes[lane]=[];slot=0;}lanes[lane].splice(slot,0,[lo,hi]);packed.push({r,blocks,lane});
  });
  const laneSize=24,pages=Math.max(1,Math.ceil(lanes.length/laneSize));mapPage=Math.max(0,Math.min(mapPage,pages-1));
  const visible=packed.filter(p=>p.lane>=mapPage*laneSize&&p.lane<(mapPage+1)*laneSize);
  const visibleSites=sites.filter(s=>x(s.position)!==null).map(s=>({...s,x:x(s.position)}));
  const outsideTargets=sites.filter(s=>s.target&&x(s.position)===null);
  q('map-target-note').hidden=!outsideTargets.length;
  q('map-target-note').textContent=outsideTargets.map(s=>{const i=regionFor(s.position);return 'Target · '+s.label+' · '+(i>=0?'Region '+(i+1):'outside the ASO binding regions');}).join('; ');
  const c=canvas.getContext('2d');if(!c)return;
  const font='-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';c.font=`11px ${font}`;
  const labels=bindingLabels(visibleSites,t=>c.measureText(t).width,left,width-right);
  const referenceY=24+Math.max(1,...labels.map(s=>s.lane+1))*20;
  const plotBottom=referenceY+128+annotationHeight,plotHeight=74,trackY=showValues?plotBottom+48:referenceY+58+annotationHeight;
  const visibleLanes=Math.max(1,Math.min(laneSize,lanes.length-mapPage*laneSize)),pitch=11,bottom=trackY+visibleLanes*pitch+12,height=bottom+34;
  const dpr=window.devicePixelRatio||1;canvas.width=width*dpr;canvas.height=height*dpr;canvas.style.width=width+'px';canvas.style.height=height+'px';c.scale(dpr,dpr);
  c.fillStyle='#fff';c.fillRect(0,0,width,height);
  const text=(value,xx,yy,size=11,color='#5b6670',align='left')=>{c.font=`${size}px ${font}`;c.fillStyle=color;c.textAlign=align;c.fillText(String(value),xx,yy);};
  const line=(x1,y1,x2,y2,color='#dbe4e8')=>{c.strokeStyle=color;c.lineWidth=1;c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke();};
  const fit=(value,limit,size=11)=>{c.font=`${size}px ${font}`;let label=String(value);if(c.measureText(label).width<=limit)return label;
   while(label&&c.measureText(label+'…').width>limit)label=label.slice(0,-1);return label?label+'…':'';};
  const features=(context.reference_features?.segments||[]).filter(s=>chromKey(s.chrom)===chromosome);
  const shownFeatures=[...new Set(features.filter(s=>axes.some(a=>s.start<=a.b&&s.end>=a.a)).map(s=>s.label))];
  q('map-feature-note').textContent=shownFeatures.length?'Transcript region in this view: '+shownFeatures.join(' · ')+'.':'CDS/UTR annotation is unavailable in this view.';
  axes.forEach(a=>line(a.x,referenceY,a.x+a.width,referenceY,'#b5c2c8'));
  if(context.reference_chrom&&chromKey(context.reference_chrom)===chromosome)
   (context.exons||[]).forEach(e=>axes.forEach(a=>{const lo=Math.max(e.start,a.a),hi=Math.min(e.end,a.b);if(hi<lo)return;const xx=x(lo),ww=x(hi+1)-xx;c.fillStyle='#c9d6dc';c.fillRect(xx,referenceY-7,Math.max(2,ww),14);
    const local=features.filter(s=>s.exon_index===e.index&&s.start<=hi&&s.end>=lo);
    local.forEach(s=>{const start=Math.max(s.start,lo),end=Math.min(s.end,hi),height=s.kind==='cds'?18:12;
     c.fillStyle=bindingFeatureColors[s.kind]||bindingFeatureColors.unknown;c.fillRect(x(start),referenceY-height/2,Math.max(1,x(end+1)-x(start)),height);});
    const kinds=[...new Set(local.map(s=>s.label))],exonName=e.label||'Annotated exon '+e.index;
    const name=(kinds.length?kinds.join(' / ')+' · ':'')+exonName,label=fit(name,ww-12,10);
    if(label)text(label,xx+ww/2,referenceY+23,10,'#48565e','center');}));
  annotationTracks.forEach((track,i)=>{const yy=referenceY+58+i*42;
   text(fit(track.label,width-left-right),left,yy-10,11,track.color);
   axes.forEach(a=>line(a.x,yy+5,a.x+a.width,yy+5,'#eef1f3'));
   track.entries.forEach(({site,x,width:w})=>{c.fillStyle=track.color;c.fillRect(x,yy,Math.max(2,w),10);
    mapAnnotationHits.push({x,y:yy,w:Math.max(2,w),h:10,site});});
  });
  visibleSites.forEach(s=>{line(s.x,referenceY-6,s.x,referenceY+8,s.target?'#15758c':'#a8b4ba');
   if(s.target){c.setLineDash([3,4]);line(s.x,referenceY+10,s.x,bottom,'#c9e0e6');c.setLineDash([]);}
   mapSiteHits.push({x:s.x-5,y:referenceY-8,w:10,h:18,site:s});
  });
  labels.forEach(s=>line(s.x,20+s.lane*20,s.x,referenceY-9,s.target?'#90bdc8':'#e1e7ea'));
  labels.forEach(s=>{const yy=16+s.lane*20;
   // Draw every label after the leaders to keep its text clear on all rows.
   c.fillStyle='#fff';c.fillRect(s.left-3,yy-12,s.width+6,16);text(s.text,s.left,yy,11,s.target?'#126477':'#5b6670');});
  if(showValues){
   let min=metric.low,max=metric.high;if(max===min){const pad=Math.max(1,Math.abs(min)*.05);min=Math.min(min,0);max+=pad;}const range=max-min;
   let title=metric.label+(metric.unit?' ('+metric.unit+')':'');c.font=`11px ${font}`;
   if(c.measureText(title).width>width-8){while(title&&c.measureText(title+'…').width>width-8)title=title.slice(0,-1);title+='…';}
   text(title,4,plotBottom-plotHeight-12);
   [0,.5,1].forEach(f=>{line(left,plotBottom-f*plotHeight,width-right,plotBottom-f*plotHeight,'#eef1f3');text(metric.tick(min+range*f),left-8,plotBottom-f*plotHeight+3,10,'#5b6670','right');});
   if(min<=0&&max>=0)line(left,plotBottom+min/range*plotHeight,width-right,plotBottom+min/range*plotHeight,'#cad4d9');
   packed.filter(({r})=>finite(value(r))).forEach(({r,blocks})=>{const [lo,hi]=blocks[0];c.fillStyle=attributeView.color(r);c.beginPath();c.arc((lo+hi)/2,plotBottom-(value(r)-min)/range*plotHeight,r.panel_rank!==null?3:2,0,2*Math.PI);c.fill();});
  }
  text(candidates.length?(hasMetric?'ASOs · '+(direction===1?'lower':'higher')+' values first':'ASO binding sites'):'No ASOs match the current filters',4,trackY-15);
  visible.forEach(({r,blocks,lane})=>{const yy=trackY+(lane-mapPage*laneSize)*pitch;
   blocks.forEach(([lo,hi])=>{const w=hi-lo;c.fillStyle=attributeView.color(r);c.fillRect(lo,yy,w,7);
    if(r.panel_rank!==null||basket.has(r.molecule)){c.strokeStyle=basket.has(r.molecule)?'#111':'#15758c';c.lineWidth=1;c.strokeRect(lo,yy,w,7);}mapHits.push({x:lo,y:yy,w,h:9,id:r.id});});});
  axes.forEach((a,i)=>{line(a.x,bottom,a.x+a.width,bottom,'#b7c4ca');if(i<axes.length-1){const xx=(a.x+a.width+axes[i+1].x)/2;
   for(const yy of [referenceY,bottom]){line(xx-5,yy+4,xx-1,yy-4,'#9daeb6');line(xx+1,yy+4,xx+5,yy-4,'#9daeb6');}
  }});
  c.font=`10px ${font}`;bindingTicks(axes,t=>c.measureText(t).width).forEach(t=>text(t.text,t.x,bottom+19,10,'#5b6670',t.align));
  q('map-description').textContent=`Chromosome ${chromosome}${context.reference_strand?' · '+context.reference_strand+' strand':''}${context.transcript?' · '+context.transcript:''}. ${region==='all'?regions.length+' binding '+(regions.length===1?'region':'regions'):'Region '+(Number(region)+1)+' of '+regions.length}. Genomic positions are 1-based.${axes.length>1?' Gaps between regions are collapsed.':''}`;
  const legend=attributeView.legend(),gradient=root.querySelector('.report-gradient');gradient.hidden=!legend.gradient;gradient.style.background=legend.background||'';q('legend').textContent=legend.text;
  q('category-legend').innerHTML=(legend.items||[]).map(item=>`<span><i style="background:${item.color}"></i>${esc(item.label)} (${item.count.toLocaleString()})</span>`).join('');
  q('map-status').textContent=`Showing ${visible.length.toLocaleString()} of ${candidates.length.toLocaleString()} matching ASOs`+(pages>1?` · Page ${mapPage+1} of ${pages}`:'');
  q('map-paging').hidden=pages===1;action('map-previous').disabled=mapPage===0;action('map-next').disabled=mapPage+1>=pages;
  canvas.setAttribute('aria-label',`ASO binding map on chromosome ${chromosome}. ${shownFeatures.join(', ')}. ${annotations.length} recorded regulatory intervals. ${visible.length} ASOs shown. Use the candidate and regulatory-site tables for details.`);
  q('map-tip').textContent='Select an ASO in the map or table to add it to your basket.';
  q('map-sites').hidden=!sites.length;
  q('map-sites-summary').textContent=`Splice sites · ${visibleSites.length} in view · ${sites.length} recorded`;
  const siteList=[...sites].sort((a,b)=>Number(b.target)-Number(a.target)||a.position-b.position).map(s=>{const i=regionFor(s.position);
   return `<tr><td>${s.target?'<strong>Target · '+esc(s.label)+'</strong>':esc(s.label)}</td><td>${i>=0?`<button type="button" class="report-button" data-map-region="${i}">Region ${i+1}</button>`:'Outside the ASO binding regions'}</td></tr>`;}).join('');
  if(mapSitesHTML!==siteList){mapSitesHTML=siteList;q('map-sites-body').innerHTML=siteList;}
 }
 function mapHit(event){const rect=q('map').getBoundingClientRect(),xx=event.clientX-rect.left,yy=event.clientY-rect.top;return mapHits.find(b=>xx>=b.x&&xx<=b.x+b.w&&yy>=b.y&&yy<=b.y+b.h);}
 q('map').addEventListener('click',event=>{const hit=mapHit(event);if(hit)toggle(hit.id);});
 q('map').addEventListener('pointermove',event=>{const hit=mapHit(event),r=hit&&byId.get(hit.id),rect=q('map').getBoundingClientRect();
  const site=mapSiteHits.find(s=>event.clientX-rect.left>=s.x&&event.clientX-rect.left<=s.x+s.w&&event.clientY-rect.top>=s.y&&event.clientY-rect.top<=s.y+s.h)?.site;
  const regulatory=mapAnnotationHits.filter(s=>event.clientX-rect.left>=s.x&&event.clientX-rect.left<=s.x+s.w&&event.clientY-rect.top>=s.y&&event.clientY-rect.top<=s.y+s.h).map(s=>s.site);
  const annotation=regulatory.slice(0,4).map(s=>`${s.label} · ${s.chrom}:${s.start.toLocaleString()}–${s.end.toLocaleString()} · ${s.evidence} · ${s.source}`).join('; ')+(regulatory.length>4?`; ${regulatory.length-4} more overlapping records in the evidence table`:'');
  q('map').style.cursor=r?'pointer':annotation?'help':'default';q('map-tip').textContent=r?`${r.name} · ${r.aso} · ${r.location} · ${attributeView.describe(r)}`:site?(site.target?'Target · ':'')+site.label:annotation||'Select an ASO in the map or table to add it to your basket.';
 });
 root.addEventListener('change',event=>{if(event.target.dataset.pick)toggle(event.target.dataset.pick);});
 root.addEventListener('click',event=>{
  const mapRegion=event.target.closest('[data-map-region]');
  if(mapRegion){q('map-region').value=mapRegion.dataset.mapRegion;mapPage=0;scheduleMap();return;}
  const recommendation=event.target.closest('[data-recommendation-pick]');
  if(recommendation){toggle(recommendation.dataset.recommendationPick);return;}
  const approach=event.target.closest('[data-approach-select]');
  if(approach){q('context').value=approach.dataset.approachSelect;contextChanged();q('context').focus({preventScroll:true});q('context').scrollIntoView({block:'nearest'});return;}
  const remove=event.target.closest('[data-remove]');if(remove){clearedBasket=null;basket.delete(remove.dataset.remove);save();renderBasket();renderTable();(q('basket-items').querySelector('[data-remove]')||action('add-top')).focus({preventScroll:true});return;}
  const th=event.target.closest('th[data-sort]');if(th){const next=th.dataset.sort;ascending=next===sort?!ascending:['order','native_rank','start','aso','off_targets'].includes(next)||next==='effect'&&contexts.get(q('context').value)?.direction==='decrease';sort=next;page=0;renderTable();return;}
  const button=event.target.closest('[data-action]');if(!button||button.disabled)return;
  const what=button.dataset.action;
  if(what==='show-recommendations'){reset();if(data.contexts.length>1){q('context').value='all';contextChanged();}q('panel-only').checked=true;sort='panel_rank';ascending=true;renderTable();const table=q('candidate-table').closest('.report-table-wrap');table.focus({preventScroll:true});table.scrollIntoView({block:'start'});return;}
  if(what==='previous'||what==='next'){page+=what==='next'?1:-1;renderTable();return;}
  if(what==='map-previous'||what==='map-next'){mapPage+=what==='map-next'?1:-1;scheduleMap();return;}
  if(what==='reset'){reset();return;}
  if(what==='empty-reset'){reset();q('search').focus({preventScroll:true});return;}
  if(what==='clear'){clearedBasket=new Map(basket);basket.clear();}
  if(what==='undo-clear'&&clearedBasket){clearedBasket.forEach((r,key)=>basket.set(key,r));clearedBasket=null;}
  if(what==='add-panel'||what==='add-recommendations'){clearedBasket=null;panelRows.forEach(r=>basket.set(r.molecule,r));}
  if(what==='add-top'){clearedBasket=null;nextMatchingRows().forEach(r=>basket.set(r.molecule,r));}
  if(['clear','undo-clear','add-panel','add-recommendations','add-top'].includes(what)){save();renderBasket();renderTable();if(what==='clear')action('undo-clear').focus({preventScroll:true});if(what==='undo-clear')action('clear').focus({preventScroll:true});return;}
  const selected=[...basket.values()];
  if(what==='order'){
   const values=[['Name','Sequence','Scale','Purification'],...selected.map((r,i)=>['ASO_'+(i+1)+' '+r.name,r.chemistry.notation,q('scale').value,q('purification').value])];
   download('ASO_order_sheet.csv',values.map(r=>r.map(csvCell).join(',')).join('\r\n'),'text/csv;charset=utf-8');
  }else if(what==='fasta')download('ASO_selection.fasta',selected.map((r,i)=>`>ASO_${i+1} ${r.name.replaceAll('\n',' ').replaceAll('\r',' ')}\n${r.aso}`).join('\n')+'\n');
  else if(what==='csv'||what==='filtered-csv'){
   const matching=new Map();filtered.forEach(r=>{if(!matching.has(r.molecule))matching.set(r.molecule,r);});
   const values=selectedMoleculeCsv(what==='filtered-csv'?[...matching.values()]:selected,byMolecule,contexts,attributeView,what==='filtered-csv'?'ASO_matching_contexts.json':'ASO_selection_contexts.json');
   download(what==='filtered-csv'?'ASO_matching_filters.csv':'ASO_selection.csv',values.map(r=>r.map(csvCell).join(',')).join('\r\n'),'text/csv;charset=utf-8');
  }else if(what==='evidence'||what==='filtered-evidence'){
   const chosen=what==='filtered-evidence'?[...new Map(filtered.map(r=>[r.molecule,r])).values()]:selected;
   download(what==='filtered-evidence'?'ASO_matching_contexts.json':'ASO_selection_contexts.json',JSON.stringify(selectedMoleculeEvidence(chosen,byMolecule,contexts,attributeView)), 'application/json');
  }
 });
 ['search','minimum','activity-minimum','panel-only','basket-only','clean-only'].forEach(role=>q(role).addEventListener('input',()=>{page=0;mapPage=0;renderTable();}));
 q('activity-context').addEventListener('change',()=>{updateActivity();attributeView.refresh();q('activity-minimum').value='';sort='oligoai';ascending=false;page=0;mapPage=0;renderTable();});
 q('context').addEventListener('change',contextChanged);q('page-size').addEventListener('change',()=>{pageSize=Number(q('page-size').value);page=0;renderTable();});
 q('map-chromosome').addEventListener('change',()=>{mapPage=0;scheduleMap();});
 q('map-region').addEventListener('change',()=>{mapPage=0;scheduleMap();});
 q('map-values').addEventListener('change',scheduleMap);
 q('top-count').addEventListener('input',updateBulkAction);
 let oldPage=0;addEventListener('beforeprint',()=>{oldPage=page;printing=true;page=0;renderTable();});addEventListener('afterprint',()=>{printing=false;page=oldPage;renderTable();});
 if(typeof ResizeObserver!=='undefined')new ResizeObserver(scheduleMap).observe(root);
 let basketHeight=0,focusFrame=0;
 function updateLayoutSpace(){root.style.setProperty('--report-basket-space',(q('basket').dataset.docked==='true'?basketHeight:0)+'px');}
 if(typeof ResizeObserver!=='undefined'){
  const observer=new ResizeObserver(entries=>{for(const entry of entries){const height=entry.borderBoxSize?.[0]?.blockSize??entry.contentRect.height;
   if(entry.target===q('basket'))basketHeight=height;else root.style.setProperty('--report-nav-space',height+'px');}updateLayoutSpace();});
  observer.observe(q('basket'));observer.observe(q('section-nav'));
 }
 function keepFocusVisible(target){
  if(!root.contains(target)||target.closest('.report-basket,.report-section-nav,.report-attribute-picker'))return;
  cancelAnimationFrame(focusFrame);focusFrame=requestAnimationFrame(()=>{
   if(document.activeElement!==target||!target.isConnected)return;
   const nav=q('section-nav'),basket=q('basket'),rect=target.getBoundingClientRect();
   const top=getComputedStyle(nav).position==='sticky'?Math.max(0,nav.getBoundingClientRect().bottom):0;
   const bottom=getComputedStyle(basket).position==='sticky'?Math.min(innerHeight,basket.getBoundingClientRect().top):innerHeight;
   if(rect.height<bottom-top-16&&(rect.top<top+8||rect.bottom>bottom-8))target.scrollIntoView({block:'center',inline:'nearest'});
  });
 }
 root.addEventListener('focusin',event=>keepFocusVisible(event.target));
 addEventListener('resize',()=>keepFocusVisible(document.activeElement));
 renderBasket();contextChanged();
}
document.querySelectorAll('.report-explorer').forEach(setup);
})();
</script>'''
EXPLORER_SCRIPT = EXPLORER_SCRIPT.replace('/* ATTRIBUTE_HELPERS */', ATTRIBUTE_SCRIPT)
EXPLORER_SCRIPT = EXPLORER_SCRIPT.replace('/* BINDING_MAP_HELPERS */', BINDING_MAP_SCRIPT)
EXPLORER_SCRIPT = EXPLORER_SCRIPT.replace('/* SELECTION_EXPORT_HELPERS */', SELECTION_EXPORT_SCRIPT)
