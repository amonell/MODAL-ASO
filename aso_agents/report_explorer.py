"""An offline candidate table, reference map and selection basket."""
import html
import json


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
    return {**data, 'encoding':'columns-v1', 'record_columns':columns, 'records':encoded}


def explorer_html(data):
    if not data or not data.get('records'):
        return ''
    esc = lambda value: html.escape(str(value if value is not None else ''), quote=True)
    facts = ''.join('<div class="report-fact"><span class="report-fact-label">'+esc(f['label'])
        +'</span><span class="report-fact-value">'+esc(f['value'])+'</span><span class="report-fact-note">'
        +esc(f.get('note'))+'</span></div>' for f in data.get('facts') or [])
    options = ''.join('<option value="'+esc(c['id'])+'">'+esc(c['label'])+'</option>' for c in data['contexts'])
    if len(data['contexts'])>1:
        options += '<option value="all">All design groups</option>'
    payload = json.dumps(pack_data(data), ensure_ascii=False, separators=(',',':'), allow_nan=False).replace('<','\\u003c').replace('&','\\u0026')
    # The initial rows make the actual results readable before scripts load.
    rows = []
    initial_context = data['contexts'][0]['id']
    for row in [r for r in data['records'] if r['context']==initial_context][:25]:
        values = ['',row.get('native_rank') if row.get('native_rank') is not None else '—',row['aso'],row['location'],
                  f"{row['effect']:+.3f}" if row['effect'] is not None else 'Unavailable',
                  f"{row['gc']*100:.0f}%" if row['gc'] is not None else 'Unavailable',
                  ' / '.join(str(int(v)) if v is not None else '—' for v in row['off_targets']),
                  row['guard']['summary']]
        rows.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>')
    return '''<section class="report-explorer" id="candidate-explorer" aria-label="Explore ASO candidates">
<div class="report-facts">'''+facts+'''</div>
<div class="report-goal"><span class="report-goal-label">What this run was asked to do</span><p data-role="objective">'''+esc(data['objective'])+'''</p></div>
<div class="report-controls"><label class="report-context">Target and approach<select data-role="context" aria-label="Target and approach">'''+options+'''</select></label>
<p class="report-sub">Choose a target to explore its predictions and ASOs.</p></div>
<section data-role="map-section" hidden><h2>Where the ASOs bind</h2><label class="report-sub" data-role="chromosome-control" hidden>Chromosome <select data-role="map-chromosome" aria-label="Map chromosome"></select></label><p class="report-map-description" data-role="map-description"></p>
<div class="report-map" tabindex="0" role="region" aria-label="Scrollable map of ASO binding sites"><canvas data-role="map" role="img" aria-label="ASO binding map; use the candidate table to select ASOs with a keyboard"></canvas></div>
<div class="report-legend"><span class="report-gradient"></span><span data-role="legend">Stronger support for the requested direction →</span></div>
<div class="report-pagination"><span data-role="map-status"></span><div><button class="report-button" data-action="map-previous">Previous rows</button><button class="report-button" data-action="map-next">Next rows</button></div></div>
<p class="report-map-tip" data-role="map-tip" aria-live="polite">Select an ASO in the map or table to add it to your basket.</p></section>
<h2>Explore all ASOs</h2>
<p class="report-sub" data-role="metric-note"></p>
<div class="report-controls"><label>Sequence, name or position<input type="search" data-role="search" placeholder="Search candidates" autocomplete="off"></label>
<label data-role="effect-control">Minimum prediction<input type="number" step="any" data-role="minimum" placeholder="Any"></label>
<label class="report-check"><input type="checkbox" data-role="panel-only">Selected panel only</label>
<label class="report-check"><input type="checkbox" data-role="clean-only">Zero exact or 1-mismatch off-targets</label>
<button type="button" class="report-button" data-action="reset">Reset filters</button></div>
<p class="report-sub"><span data-role="count" aria-live="polite">'''+str(len(data['records']))+''' results available.</span> <span data-role="screen-description">Off-targets are other sequences an ASO may also bind. Counts below show exact / 1-mismatch / 2-mismatch matches.</span></p>
<div class="report-table-wrap" tabindex="0" role="region" aria-label="ASO candidate table"><table class="report-table" data-report-native="true"><thead><tr>
<th scope="col">Select</th><th scope="col" data-sort="native_rank"><button type="button" title="Rank from the saved analysis within this target and approach">Analysis rank</button></th><th scope="col" data-sort="aso"><button type="button">ASO · 5′→3′</button></th>
<th scope="col" data-sort="start"><button type="button">Binding location</button></th><th scope="col" data-sort="effect"><button type="button">Prediction</button></th>
<th scope="col" data-sort="gc"><button type="button" title="Percentage of G and C bases">GC</button></th><th scope="col" data-sort="off_targets"><button type="button">Off-targets</button></th><th scope="col">Checks &amp; details</th></tr></thead>
<tbody>'''+''.join(rows)+'''</tbody></table></div>
<div class="report-pagination"><label>Rows per page <select data-role="page-size" aria-label="Rows per page"><option>25</option><option>50</option><option>100</option></select></label>
<span data-role="page-status" aria-live="polite"></span><div><button class="report-button" data-action="previous">Previous</button><button class="report-button" data-action="next">Next</button></div></div>
<div class="report-basket" aria-label="Selected ASO basket"><div class="report-basket-main"><span class="report-basket-count" data-role="basket-count" aria-live="polite">No ASOs selected</span>
<div class="report-basket-actions"><button class="report-button" data-action="add-panel">Add panel</button><button class="report-button" data-action="add-top">Add first 10 shown</button>
<button class="report-button report-button-primary" data-action="csv" disabled>Download CSV</button><button class="report-button" data-action="fasta" disabled>FASTA</button><button class="report-button" data-action="clear" disabled>Clear</button></div></div>
<details><summary>Selected ASOs and ordering options</summary><ul class="report-basket-items" data-role="basket-items"></ul>
<div class="report-order-fields"><label>Number to add<input type="number" data-role="top-count" min="1" max="10000" value="10"></label><label>Scale<input data-role="scale" placeholder="Specify with supplier"></label><label>Purification<input data-role="purification" placeholder="Specify with supplier"></label></div>
<p class="report-sub">The order sheet uses each ASO’s saved ordering notation exactly. CSV includes every recorded design context for your selected molecules.</p>
<p class="report-notice" data-role="order-notice" hidden></p><p class="report-sub" data-role="storage-notice" hidden></p>
<button class="report-button" data-action="order" disabled>Download order sheet</button></details></div>
<noscript><p class="report-explorer-noscript">The first 25 records are shown here. Enable JavaScript for search, the map, selection and exports. Saved analysis and file links remain below.</p></noscript>
<script type="application/json" class="report-explorer-data">'''+payload+'''</script></section>'''


EXPLORER_SCRIPT = r'''<script>
(()=>{'use strict';
const esc=value=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const fmt=value=>finite(value)?(value>0?'+':'')+value.toFixed(3):'Unavailable';
const ramp=value=>{if(!finite(value)||value<=0)return '#e9eded';const a=Math.pow(Math.min(value,1),.7);return `rgb(${Math.round(226-190*a)},${Math.round(234-106*a)},${Math.round(234-106*a)})`;};
const csvCell=value=>{let text=String(value??'');if(typeof value==='string'&&!['+','-'].includes(text)&&'=+@-'.includes(text.trimStart().charAt(0))&&text.trim())text="'"+text;return '"'+text.replaceAll('"','""')+'"';};
const download=(name,text,type)=>{const url=URL.createObjectURL(new Blob([text],{type:type||'text/plain;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();setTimeout(()=>{link.remove();URL.revokeObjectURL(url);},1000);};
function setup(root){
 if(root.dataset.initialized)return;root.dataset.initialized='true';
 const packet=root.querySelector('.report-explorer-data');if(!packet)return;
 let data;try{data=JSON.parse(packet.textContent);}catch{root.insertAdjacentHTML('afterbegin','<p class="report-notice">Interactive tools could not load. The saved table and analysis remain available below.</p>');return;}
 if(data.encoding==='columns-v1')data.records=data.records.map(values=>{
  const row={};data.record_columns.forEach((column,i)=>row[column.name]=column.values?column.values[values[i]]:values[i]);
  row.chemistry={...row.chemistry,notation:row.notation};delete row.notation;return row;
 });
 const q=role=>root.querySelector(`[data-role="${role}"]`),action=name=>root.querySelector(`[data-action="${name}"]`);
 const rows=data.records,byId=new Map(rows.map(r=>[r.id,r])),contexts=new Map(data.contexts.map(c=>[c.id,c]));
 const basket=new Map(),storageKey='modal-aso-selection:'+data.id;
 let page=0,mapPage=0,pageSize=25,sort='effect',ascending=false,filtered=[],mapHits=[],mapFrame=0,printing=false;
 try{const saved=JSON.parse(localStorage.getItem(storageKey)||'[]');if(Array.isArray(saved))saved.forEach(id=>{const r=byId.get(id);if(r)basket.set(r.molecule,r);});}catch{q('storage-notice').textContent='Selections stay available while this report is open. Browser storage is unavailable.';q('storage-notice').hidden=false;}
 const scopeRows=()=>q('context').value==='all'?rows:rows.filter(r=>r.context===q('context').value);
 function visible(){
  const query=q('search').value.trim().toUpperCase(),minimum=q('minimum').value.trim();
  let result=scopeRows().filter(r=>(!q('panel-only').checked||r.panel_rank!==null)&&
   (!q('clean-only').checked||(r.off_target_complete&&r.off_targets[0]===0&&r.off_targets[1]===0))&&
   (!minimum||(finite(r.effect)&&r.effect>=Number(minimum)))&&
   (!query||[r.aso,r.name,r.location,contexts.get(r.context)?.label].some(v=>String(v||'').toUpperCase().includes(query))));
  const multi=q('context').value==='all';
  return result.sort((a,b)=>{
   if(multi&&a.context!==b.context)return data.contexts.findIndex(c=>c.id===a.context)-data.contexts.findIndex(c=>c.id===b.context);
   let x=sort==='off_targets'?a.off_targets[0]:a[sort],y=sort==='off_targets'?b.off_targets[0]:b[sort];
   if(x===null||x===undefined)return y===null||y===undefined?a.order-b.order:1;if(y===null||y===undefined)return -1;
   const delta=typeof x==='string'?x.localeCompare(y):x-y;return (ascending?delta:-delta)||a.order-b.order;
  });
 }
 function renderTable(){
  filtered=visible();const total=filtered.length,size=printing?Math.max(1,total):pageSize;
  page=Math.max(0,Math.min(page,Math.max(0,Math.ceil(total/size)-1)));
  const shown=filtered.slice(page*size,(page+1)*size),scoped=scopeRows();
  const maxima=new Map();scoped.forEach(r=>{if(finite(r.effect))maxima.set(r.context,Math.max(maxima.get(r.context)||.000001,r.effect));});
  root.querySelector('tbody').innerHTML=shown.map(r=>{
   const max=maxima.get(r.context)||.000001;
   const color=ramp(finite(r.effect)?r.effect/max:null),white=finite(r.effect)&&r.effect/max>.55;
   const raw=r.predictions.length?`<dt>Deletion / replacement prediction</dt><dd>${r.predictions.map(fmt).join(' / ')}</dd>`:'';
   const checks=r.guard.details.map(g=>`<p>${esc(g.text)}${g.reason?'<br>'+esc(g.reason):''}</p>`).join('');
   const chemistry=esc(r.chemistry.detail||r.chemistry.label);
   const screen=r.off_targets.map(v=>finite(v)?v.toLocaleString():'—').join(' / ');
   return `<tr data-record="${esc(r.id)}" data-panel="${r.panel_rank!==null}" aria-selected="${basket.has(r.molecule)}"><td><input type="checkbox" data-pick="${esc(r.id)}" aria-label="Select ${esc(r.name)}" ${basket.has(r.molecule)?'checked':''}></td><td class="report-number">${r.native_rank??'—'}</td>
   <td><span class="report-sequence">${esc(r.aso)}</span>${r.panel_rank!==null?` <span class="report-badge report-badge-panel">Panel ${r.panel_rank}</span>`:''}<div class="report-sub">${esc(r.chemistry.label)}</div></td>
   <td><span class="report-sub">${esc(r.location)}</span>${q('context').value==='all'?'<div class="report-sub">'+esc(contexts.get(r.context)?.label)+'</div>':''}</td>
   <td class="report-number"><span class="report-effect" style="background:${color};color:${white?'#fff':'#1b1f23'}">${fmt(r.effect)}</span></td><td class="report-number">${finite(r.gc)?(r.gc*100).toFixed(0)+'%':'—'}</td>
   <td><span class="report-badge">${screen}</span><div class="report-sub">${esc(r.off_target_unit)}${r.off_target_complete?'':r.off_target_status==='unavailable'?' · search result missing':r.off_target_status==='failed'?' · search failed':' · search incomplete'}</div></td>
   <td><details><summary>${['passed','review','incomplete'].includes(r.guard.status)?esc(r.guard.summary):'View ASO details'}</summary><div class="report-check-text">${checks||esc(r.guard.summary)}<dl><dt>Selection</dt><dd>${esc(r.selection_note)}</dd><dt>Chemistry</dt><dd>${chemistry}</dd>${raw}<dt>Target and approach</dt><dd>${esc(contexts.get(r.context)?.label)}</dd><dt>Saved rank</dt><dd>${r.native_rank??'Unavailable'}</dd><dt>Ordering notation</dt><dd><code>${esc(r.chemistry.notation||'Unavailable')}</code></dd></dl></div></details></td></tr>`;
  }).join('')||'<tr><td colspan="8" class="report-empty">No ASOs match these filters.</td></tr>';
  const molecules=new Set(filtered.map(r=>r.molecule)).size;
  q('count').textContent=`${total.toLocaleString()} of ${scoped.length.toLocaleString()} results · ${molecules.toLocaleString()} ASOs.`;
  q('page-status').textContent=total?`${(page*size+1).toLocaleString()}–${Math.min((page+1)*size,total).toLocaleString()} of ${total.toLocaleString()}`:'No matching records';
  action('previous').disabled=page===0;action('next').disabled=(page+1)*size>=total;
  root.querySelectorAll('th[data-sort]').forEach(th=>th.setAttribute('aria-sort',th.dataset.sort===sort?(ascending?'ascending':'descending'):'none'));
  scheduleMap();
 }
 function save(){try{localStorage.setItem(storageKey,JSON.stringify([...basket.values()].map(r=>r.id)));}catch{q('storage-notice').hidden=false;q('storage-notice').textContent='Selections stay available while this report is open. Browser storage is unavailable.';}}
 function renderBasket(){
  const n=basket.size;q('basket-count').textContent=n?`${n.toLocaleString()} ASO${n===1?'':'s'} selected`:'No ASOs selected';
  ['csv','fasta','clear'].forEach(a=>action(a).disabled=!n);
  const selected=[...basket.values()],missing=selected.filter(r=>!r.chemistry.notation).length;
  action('order').disabled=!n||missing>0;
  q('basket-items').innerHTML=selected.slice(0,100).map(r=>`<li><span><b>${esc(r.name)}</b> · ${esc(r.aso)} · ${esc(r.chemistry.label)}</span><button class="report-button" data-remove="${esc(r.molecule)}" aria-label="Remove ${esc(r.name)}">Remove</button></li>`).join('')+(n>100?`<li>${n-100} more selections are included in exports.</li>`:'');
  const issues=[...new Set(selected.flatMap(r=>r.chemistry.issues||[]))];
  q('order-notice').hidden=!issues.length;q('order-notice').textContent=(missing?`${missing} selections need ordering notation. `:'')+issues.join(' ');
 }
 function toggle(id){const row=byId.get(id);if(!row)return;if(basket.has(row.molecule))basket.delete(row.molecule);else basket.set(row.molecule,row);save();renderBasket();renderTable();}
 function reset(){q('search').value='';q('minimum').value='';q('panel-only').checked=false;q('clean-only').checked=false;page=0;mapPage=0;renderTable();}
 function contextChanged(){
  const context=contexts.get(q('context').value),hasEffects=scopeRows().some(r=>finite(r.effect));
  q('objective').textContent=context?.objective||data.objective;
  q('metric-note').textContent=context?.metric_note||'Each design group keeps its own target, prediction units and recorded ranking.';
  root.classList.toggle('report-no-predictions',!hasEffects);
  const hasScreen=data.kind!=='chemistry'||scopeRows().some(r=>r.off_targets.some(finite)||r.off_target_status!=='unavailable');
  root.classList.toggle('report-no-screening',!hasScreen);q('clean-only').closest('label').hidden=!hasScreen;q('screen-description').hidden=!hasScreen;
  root.classList.toggle('report-no-locations',!scopeRows().some(r=>r.intervals.length));
  root.classList.toggle('report-no-gc',!scopeRows().some(r=>finite(r.gc)));
  root.classList.toggle('report-no-ranking',!scopeRows().some(r=>finite(r.native_rank)));
  const hasPanel=scopeRows().some(r=>r.panel_rank!==null);
  q('panel-only').closest('label').hidden=!hasPanel;if(!hasPanel)q('panel-only').checked=false;
  action('add-panel').hidden=!rows.some(r=>r.panel_rank!==null);
  if(data.contexts.length===1)q('context').closest('.report-controls').hidden=true;
  q('effect-control').hidden=!hasEffects||!context;q('minimum').value='';sort=hasEffects&&context?'effect':'order';ascending=sort==='order';page=0;mapPage=0;renderTable();
 }
 function scheduleMap(){cancelAnimationFrame(mapFrame);mapFrame=requestAnimationFrame(drawMap);}
 function drawMap(){
  const context=contexts.get(q('context').value),section=q('map-section');
  let candidates=context?filtered.filter(r=>r.intervals.length):[];
  const chromKey=value=>String(value).replace(/^chr/,'');
  const chromosomes=[...new Set(candidates.flatMap(r=>r.intervals.map(p=>chromKey(p.chrom))))];
  section.hidden=!context||!candidates.length;
  if(section.hidden){mapHits=[];return;}
  let chromosome=q('map-chromosome').value;if(!chromosomes.includes(chromosome))chromosome=chromosomes[0];
  q('map-chromosome').innerHTML=chromosomes.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join('');q('map-chromosome').value=chromosome;
  q('chromosome-control').hidden=chromosomes.length<2;
  candidates=candidates.filter(r=>r.intervals.every(p=>chromKey(p.chrom)===chromosome));
  if(!candidates.length){section.hidden=true;mapHits=[];return;}
  const all=scopeRows().filter(r=>r.intervals.length&&r.intervals.every(p=>chromKey(p.chrom)===chromosome));
  const sites=(context.sites||[]).filter(s=>chromKey(s.chrom)===chromosome);
  const intervals=all.flatMap(r=>r.intervals.map(p=>[Math.max(1,p.start-12),p.end+12])).concat(sites.map(s=>[Math.max(1,s.position-12),s.position+12])).sort((a,b)=>a[0]-b[0]);
  const segments=[];intervals.forEach(p=>{const last=segments.at(-1);if(last&&p[0]<=last[1]+70)last[1]=Math.max(last[1],p[1]);else segments.push(p.slice());});
  const canvas=q('map'),width=Math.max(700,Math.floor(canvas.parentElement.clientWidth-24)),left=54,right=20,gap=Math.min(28,width/Math.max(3,segments.length*3));
  let total=segments.reduce((s,p)=>s+p[1]-p[0]+1,0),available=width-left-right-gap*(segments.length-1),offset=left;
  const axes=segments.map(([a,b])=>{const length=(b-a+1)/total*available;const result={a,b,x:offset,width:length};offset+=length+gap;return result;});
  const axisFor=p=>axes.find(a=>p>=a.a&&p<=a.b),x=p=>{const a=axisFor(p);return a?a.x+(p-a.a)/(a.b-a.a+1)*a.width:null;};
  candidates.sort((a,b)=>(finite(b.effect)?b.effect:-Infinity)-(finite(a.effect)?a.effect:-Infinity)||(a.native_rank??a.order)-(b.native_rank??b.order));
  const lanes=[],packed=[];
  candidates.forEach(r=>{const blocks=r.intervals.map(p=>[x(p.start),x(p.end)]).filter(p=>p.every(v=>v!==null));if(!blocks.length)return;
   const lo=Math.min(...blocks.map(b=>b[0])),hi=Math.max(...blocks.map(b=>b[1]));let lane=0;
   while(lane<lanes.length&&lanes[lane].some(b=>!(hi+2<b[0]||lo>b[1]+2)))lane++;
   if(!lanes[lane])lanes[lane]=[];lanes[lane].push([lo,hi]);packed.push({r,blocks,lane});
  });
  const laneSize=35;mapPage=Math.max(0,Math.min(mapPage,Math.ceil(lanes.length/laneSize)-1));
  const hasEffects=candidates.some(r=>finite(r.effect));
  const visibleLanes=Math.min(laneSize,lanes.length-mapPage*laneSize),trackY=hasEffects?220:115,height=trackY+visibleLanes*6+52;
  const dpr=window.devicePixelRatio||1;canvas.width=width*dpr;canvas.height=height*dpr;canvas.style.width=width+'px';canvas.style.height=height+'px';
  const c=canvas.getContext('2d');if(!c)return;c.scale(dpr,dpr);c.fillStyle='#fff';c.fillRect(0,0,width,height);
  const text=(value,xx,yy,size=10,color='#5b6670',align='left')=>{c.font=`${size}px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif`;c.fillStyle=color;c.textAlign=align;c.fillText(String(value),xx,yy);};
  const line=(x1,y1,x2,y2,color='#dbe4e8')=>{c.strokeStyle=color;c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke();};
  text(context.transcript||'Recorded binding region',4,12,11);
  axes.forEach(a=>line(a.x,33,a.x+a.width,33,'#b5c2c8'));
  if(context.reference_chrom&&chromKey(context.reference_chrom)===chromosome)
   (context.exons||[]).forEach(e=>axes.forEach(a=>{const lo=Math.max(e.start,a.a),hi=Math.min(e.end,a.b);if(hi<lo)return;const xx=x(lo),ww=Math.max(2,x(hi)-xx);c.fillStyle='#c9d6dc';c.fillRect(xx,26,ww,14);if(ww>36)text('exon '+e.index,xx+ww/2,23,9,'#48565e','center');}));
  sites.forEach((s,i)=>{const xx=x(s.position);if(xx===null)return;const yy=59+(i%3)*13;line(xx,40,xx,yy-7,s.target?'#15758c':'#a8b4ba');text(s.label+(s.target?' · target':''),Math.min(width-10,Math.max(10,xx+5)),yy,9,s.target?'#15758c':'#5b6670',xx>width-220?'right':'left');});
  const max=all.reduce((v,r)=>finite(r.effect)?Math.max(v,r.effect):v,.000001);
  const min=all.reduce((v,r)=>finite(r.effect)?Math.min(v,r.effect):v,0),range=max-min;
  if(hasEffects){
  text('PREDICTED SUPPORT FOR THE REQUESTED DIRECTION',4,107,10);
  const plotBottom=188,plotHeight=65;
  [0,.5,1].forEach(f=>{line(left,plotBottom-f*plotHeight,width-right,plotBottom-f*plotHeight,'#eef1f3');if(candidates.some(r=>finite(r.effect)))text((min+range*f).toFixed(2),left-7,plotBottom-f*plotHeight+3,9,'#5b6670','right');});
  line(left,plotBottom+min/range*plotHeight,width-right,plotBottom+min/range*plotHeight,'#cad4d9');
  candidates.filter(r=>finite(r.effect)).forEach(r=>{const p=r.intervals[0],xx=x((p.start+p.end)/2);if(xx===null)return;c.fillStyle=ramp(r.effect/max);c.beginPath();c.arc(xx,plotBottom-(r.effect-min)/range*plotHeight,r.panel_rank!==null?3:2,0,2*Math.PI);c.fill();});
  }
  text(candidates.some(r=>finite(r.effect))?'ASOs · stronger predictions in the top rows':'ASOs · saved order',4,trackY-10,10);
  mapHits=[];packed.forEach(({r,blocks,lane})=>{if(lane<mapPage*laneSize||lane>=(mapPage+1)*laneSize)return;const yy=trackY+(lane-mapPage*laneSize)*6;
   blocks.forEach(([x0,x1])=>{const w=Math.max(2,x1-x0);c.fillStyle=ramp(finite(r.effect)?r.effect/max:null);c.fillRect(x0,yy,w,4.8);
    if(r.panel_rank!==null||basket.has(r.molecule)){c.strokeStyle=basket.has(r.molecule)?'#111':'#15758c';c.lineWidth=1;c.strokeRect(x0,yy,w,4.8);}mapHits.push({x:x0,y:yy,w,h:6,id:r.id});});});
  const bottom=trackY+visibleLanes*6+12;
  axes.forEach((a,i)=>{line(a.x,bottom,a.x+a.width,bottom,'#b7c4ca');if(a.width>75||i===0)text(a.a.toLocaleString(),a.x,bottom+14,9);if(a.width>140)text(a.b.toLocaleString(),a.x+a.width,bottom+14,9,'#5b6670','right');
   if(i<axes.length-1){c.setLineDash([3,3]);line(a.x+a.width+gap/2,25,a.x+a.width+gap/2,bottom,'#b3bfc5');c.setLineDash([]);if(axes.length<5)text((axes[i+1].a-a.b-1).toLocaleString()+' bp omitted',a.x+a.width+gap/2,bottom+31,9,'#5b6670','center');}});
  q('map-description').textContent=`Chromosome ${chromosome} · 1-based genomic coordinates${context.reference_strand?' · '+context.reference_strand+' strand':''}. ${context.transcript?'Exon numbers follow '+context.transcript+'. ':''}${axes.length>1?(axes.length-1)+(axes.length===2?' untiled gap is collapsed. ':' untiled gaps are collapsed. '):''}The table retains every binding interval.`;
  root.querySelector('.report-gradient').hidden=!hasEffects;
  q('legend').textContent=hasEffects?'Grey: zero or opposing predictions. Teal: stronger support →':'Grey blocks show recorded binding positions';
  q('map-status').textContent=`ASO lanes ${mapPage*laneSize+1}–${Math.min((mapPage+1)*laneSize,lanes.length)} of ${lanes.length}`;
  action('map-previous').disabled=mapPage===0;action('map-next').disabled=(mapPage+1)*laneSize>=lanes.length;
 }
 function mapHit(event){const rect=q('map').getBoundingClientRect(),xx=event.clientX-rect.left,yy=event.clientY-rect.top;return mapHits.find(b=>xx>=b.x&&xx<=b.x+b.w&&yy>=b.y&&yy<=b.y+b.h);}
 q('map').addEventListener('click',event=>{const hit=mapHit(event);if(hit)toggle(hit.id);});
 q('map').addEventListener('pointermove',event=>{const hit=mapHit(event),r=hit&&byId.get(hit.id);q('map').style.cursor=r?'pointer':'default';q('map-tip').textContent=r?`${r.name} · ${r.aso} · ${r.location} · ${fmt(r.effect)}`:'Select an ASO in the map or table to add it to your basket.';});
 root.addEventListener('change',event=>{if(event.target.dataset.pick)toggle(event.target.dataset.pick);});
 root.addEventListener('click',event=>{
  const remove=event.target.closest('[data-remove]');if(remove){basket.delete(remove.dataset.remove);save();renderBasket();renderTable();return;}
  const th=event.target.closest('th[data-sort]');if(th){const next=th.dataset.sort;ascending=next===sort?!ascending:['order','native_rank','start','aso','off_targets'].includes(next);sort=next;page=0;renderTable();return;}
  const button=event.target.closest('[data-action]');if(!button||button.disabled)return;
  const what=button.dataset.action;
  if(what==='previous'||what==='next'){page+=what==='next'?1:-1;renderTable();return;}
  if(what==='map-previous'||what==='map-next'){mapPage+=what==='map-next'?1:-1;scheduleMap();return;}
  if(what==='reset'){reset();return;}
  if(what==='clear')basket.clear();
  if(what==='add-panel')rows.filter(r=>r.panel_rank!==null).forEach(r=>basket.set(r.molecule,r));
  if(what==='add-top')filtered.slice(0,Math.max(1,Number(q('top-count').value)||10)).forEach(r=>basket.set(r.molecule,r));
  if(['clear','add-panel','add-top'].includes(what)){save();renderBasket();renderTable();return;}
  const selected=[...basket.values()];
  if(what==='order'){
   const values=[['Name','Sequence','Scale','Purification'],...selected.map((r,i)=>['ASO_'+(i+1)+' '+r.name,r.chemistry.notation,q('scale').value,q('purification').value])];
   download('ASO_order_sheet.csv',values.map(r=>r.map(csvCell).join(',')).join('\r\n'),'text/csv;charset=utf-8');
  }else if(what==='fasta')download('ASO_selection.fasta',selected.map((r,i)=>`>ASO_${i+1} ${r.name.replaceAll('\n',' ').replaceAll('\r',' ')}\n${r.aso}`).join('\n')+'\n');
  else if(what==='csv'){
   const columns=['name','antisense_5to3','ordering_notation','chemistry','binding_intervals_1based','strand','design_group','program_id','job_id','response_site','native_rank','display_order','prediction_toward_goal','deletion_prediction','replacement_prediction','gc_fraction','off_target_exact','off_target_1mm','off_target_2mm','off_target_unit','off_target_status','splice_checks','panel_order','selection_note'];
   const values=rows.filter(r=>basket.has(r.molecule)).map(r=>[r.name,r.aso,r.chemistry.notation,r.chemistry.label+'; '+r.chemistry.detail,r.location,r.strand,contexts.get(r.context)?.label,r.origin.program,r.origin.job,r.origin.route,r.native_rank,r.order,r.effect,...[0,1].map(i=>r.predictions[i]??''),r.gc,...r.off_targets,r.off_target_unit,r.off_target_status,r.guard.summary,r.panel_rank,r.selection_note]);
   download('ASO_selection.csv',[columns,...values].map(r=>r.map(csvCell).join(',')).join('\r\n'),'text/csv;charset=utf-8');
  }
 });
 ['search','minimum','panel-only','clean-only'].forEach(role=>q(role).addEventListener('input',()=>{page=0;mapPage=0;renderTable();}));
 q('context').addEventListener('change',contextChanged);q('page-size').addEventListener('change',()=>{pageSize=Number(q('page-size').value);page=0;renderTable();});
 q('map-chromosome').addEventListener('change',()=>{mapPage=0;scheduleMap();});
 q('top-count').addEventListener('input',()=>{action('add-top').textContent='Add first '+(Number(q('top-count').value)||10)+' shown';});
 let oldPage=0;addEventListener('beforeprint',()=>{oldPage=page;printing=true;page=0;renderTable();});addEventListener('afterprint',()=>{printing=false;page=oldPage;renderTable();});
 if(typeof ResizeObserver!=='undefined')new ResizeObserver(scheduleMap).observe(root);
 renderBasket();contextChanged();
}
document.querySelectorAll('.report-explorer').forEach(setup);
})();
</script>'''
