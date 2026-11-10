"""Offline, typed attribute exploration shared by native ASO reports."""

ATTRIBUTE_HTML = '''
<section class="report-attributes" data-role="attribute-explorer" aria-label="ASO attributes and filters" hidden>
<div class="report-attribute-toolbar"><div><h2>Explore ASO attributes</h2><p class="report-sub">Color the plot, compare distributions, and combine filters.</p></div>
<button type="button" class="report-button" data-attribute-action="add">Add attribute filter</button>
<button type="button" class="report-button" data-action="filtered-csv">Download matching ASOs</button></div>
<div class="report-color-controls"><span>Color by</span><div data-role="attribute-quick" class="report-attribute-quick"></div>
<button type="button" class="report-button" data-attribute-action="color"><span data-role="attribute-color-label">Choose an attribute</span> ▾</button></div>
<p class="report-sub" data-role="attribute-color-description"></p>
<div data-role="attribute-color-histogram" class="report-color-histogram"></div>
<div class="report-filter-heading"><p data-role="attribute-summary" aria-live="polite">All ASOs are included.</p><button type="button" class="report-button" data-attribute-action="clear" hidden>Clear attribute filters</button></div>
<div data-role="attribute-filters" class="report-attribute-filters"></div>
<dialog class="report-attribute-picker" aria-label="Choose a stored ASO attribute">
<div class="report-attribute-toolbar"><h2 data-role="attribute-picker-title">Choose an attribute</h2><button type="button" class="report-button" data-attribute-action="close" aria-label="Close attribute chooser">Close</button></div>
<label>Find an attribute<input type="search" data-role="attribute-search" placeholder="Activity, conservation, motif, chemistry…" autocomplete="off"></label>
<p class="report-sub" data-role="attribute-search-count"></p>
<select size="9" data-role="attribute-options" aria-label="Stored ASO attributes"></select>
<p class="report-sub" data-role="attribute-picker-description"></p>
<button type="button" class="report-button report-button-primary" data-attribute-action="choose">Use attribute</button>
</dialog></section>
'''


ATTRIBUTE_SCRIPT = r'''
function setupAttributes({root,rows,attributes,scopeRows,baseRows,activityFor,changed}){
 const q=role=>root.querySelector(`[data-role="${role}"]`),dialog=root.querySelector('.report-attribute-picker');
 const rowIndex=new Map(rows.map((r,i)=>[r.id,i])),catalog=new Map((attributes||[]).map(a=>[a.id,a]));
 const filters=new Map(),domains=new Map(),categoryColors=new Map(),categoryChoices=new Map(),knownValues=new Map(),decodedArrays=new WeakMap();let colorId='',pickerMode='color',histogramOpen=true;
 const missingColor='#b9c0c8',palette=['#16788c','#b16b2e','#8863aa','#44865b','#bc5d82','#5279b6','#a38c27','#755b4b','#529b99','#8b738a'];
 const format=(value,unit='')=>finite(value)?value.toLocaleString(undefined,{maximumSignificantDigits:5})+(unit?' '+unit:''):'Unavailable';
 const label=(a,v)=>Object.hasOwn(a.labels||{},String(v))?a.labels[String(v)]:(typeof v==='boolean'?(v?'Yes':'No'):String(v));
 function values(a,row){
  if(a.dynamic==='oligoai'){const activity=activityFor(row);return [activity.missing?null:activity.value];}
  while(a.value_ref)a=catalog.get(a.value_ref);
  let value=Object.hasOwn(a,'constant')?a.constant:a.values?.[rowIndex.get(row.id)];
  if(a.dictionary)value=a.dictionary[value];
  if(a.atoms){
   if(Array.isArray(value)){
    if(!decodedArrays.has(value))decodedArrays.set(value,value.map(v=>v===null?null:a.atoms[v]));
    return decodedArrays.get(value);
   }
   value=value===null||value===undefined?null:a.atoms[value];
  }
  return Array.isArray(value)?value:[value??null];
 }
 function known(a,row){
  const active=a.id===colorId||filters.has(a.id);let cache=knownValues.get(a.id);
  if(active&&cache?.has(row.id))return cache.get(row.id);
  const raw=values(a,row),valid=v=>v!==null&&v!==undefined&&(a.kind!=='number'||finite(v));
  const result=raw.every(valid)?raw:raw.filter(valid);
  if(active){if(!cache){cache=new Map();knownValues.set(a.id,cache);}cache.set(row.id,result);}
  return result;
 }
 const numeric=(a,row)=>known(a,row).reduce((low,v)=>low===null?v:Math.min(low,v),null);
 function domain(a){
  if(domains.has(a.id))return domains.get(a.id);
  let low=Infinity,high=-Infinity,count=0,multiple=!!a.multiple,integer=true;
  for(const row of scopeRows()){
   const present=known(a,row);count+=present.length;multiple=multiple||present.length>1;
   if(a.kind==='number')for(const v of present){low=Math.min(low,v);high=Math.max(high,v);integer=integer&&Number.isInteger(v);}
   else if(count)break;
  }
  const exists=count>0;
  if(a.kind==='number'){
   if(!exists){low=finite(a.low)?a.low:0;high=finite(a.high)?a.high:1;}else{if(finite(a.low))low=Math.min(low,a.low);if(finite(a.high))high=Math.max(high,a.high);}
  }
  const result={low,high,exists,multiple,integer};
  domains.set(a.id,result);return result;
 }
 function categoryKey(a,row){return JSON.stringify(known(a,row));}
 function categoryLabel(a,key){
  const values=JSON.parse(key),parts=[];let last,count=0;
  for(const v of values){if(v===last){count++;parts[parts.length-1]=label(a,v)+' × '+count;}else{parts.push(label(a,v));last=v;count=1;}}
  return parts.slice(0,5).join(' · ')+(parts.length>5?' · '+(parts.length-5)+' more values':'');
 }
 function categoryPalette(a){
  if(categoryColors.has(a.id))return categoryColors.get(a.id);
  const keys=[...new Set(scopeRows().filter(r=>known(a,r).length).map(r=>categoryKey(a,r)))].sort();
  const colors=new Map(keys.map((key,i)=>[key,i<palette.length?palette[i]:`hsl(${(i*137.508)%360} 48% 44%)`]));
  categoryColors.set(a.id,colors);return colors;
 }
 const blend=(a,b,t)=>`rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`;
 function color(row){
  const a=catalog.get(colorId);if(!a)return missingColor;
  if(a.kind!=='number')return categoryPalette(a).get(categoryKey(a,row))||missingColor;
  const v=numeric(a,row),d=domain(a);if(v===null)return missingColor;
  if(d.low<0&&d.high>0)return v<0?blend([234,244,246],[167,94,42],v/d.low):blend([234,244,246],[18,104,127],v/d.high);
  let t=d.high===d.low?.7:(v-d.low)/(d.high-d.low);if(a.direction==='lower')t=1-t;
  return blend([219,237,244],[18,104,127],Math.max(0,Math.min(1,t)));
 }
 function matches(row,except){
  for(const [id,f] of filters){
   if(id===except)continue;const a=catalog.get(id);if(!a)continue;
   const raw=values(a,row),v=known(a,row),missing=raw.some(x=>x===null||x===undefined||(a.kind==='number'&&!finite(x)));
   if(!v.length){if(f.includeMissing)continue;return false;}
   if(missing&&!f.includeMissing&&f.mode==='all')return false;
   const match=x=>a.kind==='number'?x>=f.low&&x<=f.high:(!f.categories.size||f.categories.has(String(x)))&&(!f.text||label(a,x).toLocaleLowerCase().includes(f.text.toLocaleLowerCase()));
   if(!(f.mode==='any'?v.some(match):v.every(match)))return false;
  }
  return true;
 }
 function histogram(a,population,filter){
  const d=domain(a),width=360,height=62,baseline=48;
  if(!d.exists)return '<p class="report-sub">No values are recorded for this target.</p>';
  let bars=[],maximum=1,missing=0,observations=0;
  if(a.kind==='number'){
   const n=d.low===d.high?1:24,bins=Array(n).fill(0),selected=Array(n).fill(0),range=d.high-d.low||1;
   population.forEach(r=>{const v=known(a,r);if(!v.length)missing++;v.forEach(value=>{const i=Math.min(n-1,Math.max(0,Math.floor((value-d.low)/range*n)));bins[i]++;if(!filter||(value>=filter.low&&value<=filter.high))selected[i]++;observations++;});});
   maximum=Math.max(1,...bins);
   bars=bins.map((count,i)=>{const lo=d.low+range*i/n,hi=d.low===d.high?d.high:d.low+range*(i+1)/n;
    const x=i*width/n,w=Math.max(width/n-1,1),h=count/maximum*42,chosenHeight=selected[i]/maximum*42;
    const title=`${esc(format(lo,a.unit))}–${esc(format(hi,a.unit))}: ${count.toLocaleString()} ${d.multiple?'values':'ASOs'}${filter?' · '+selected[i].toLocaleString()+' in range':''}`;
    return `<g><title>${title}</title><rect x="${x}" y="${baseline-h}" width="${w}" height="${h}" fill="#dce4e8"/><rect x="${x}" y="${baseline-chosenHeight}" width="${w}" height="${chosenHeight}" fill="#16788c"/></g>`;});
   return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Distribution of ${esc(a.label)}; ${observations} recorded ${d.multiple?'values':'ASOs'}">${bars.join('')}<line x1="0" y1="49" x2="360" y2="49" stroke="#cbd5da"/><text x="0" y="61">${esc(format(d.low,a.unit))}</text><text x="360" y="61" text-anchor="end">${esc(format(d.high,a.unit))}</text></svg><p class="report-sub">${observations.toLocaleString()} ${d.multiple?'recorded values':'ASOs with values'} · ${missing.toLocaleString()} unavailable${d.multiple?'. Each ASO can contribute several values.':''}</p>`;
  }
  const counts=new Map();population.forEach(r=>{const v=[...new Set(known(a,r).map(String))];if(!v.length)missing++;v.forEach(value=>counts.set(value,(counts.get(value)||0)+1));});
  const entries=[...counts].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));maximum=entries.reduce((max,e)=>Math.max(max,e[1]),1);
  const chosen=filter?.categorySearch?.toLocaleLowerCase()||'';
  const visible=entries.filter(([v])=>!chosen||label(a,v).toLocaleLowerCase().includes(chosen)).slice(0,12);
  return '<div class="report-category-bars">'+visible.map(([value,count])=>`<div><span title="${esc(label(a,value))}">${esc(label(a,value))}</span><span class="report-category-track"><i style="width:${count/maximum*100}%"></i></span><span>${count.toLocaleString()}</span></div>`).join('')+'</div><p class="report-sub">'+entries.length.toLocaleString()+' values · '+missing.toLocaleString()+' unavailable'+(entries.length>12?' · Showing the most frequent values.':'')+'</p>';
 }
 function options(){
  const search=q('attribute-search').value.toLocaleLowerCase(),previous=q('attribute-options').value;
  const matches=[...catalog.values()].filter(a=>[a.label,a.group,a.description].join(' ').toLocaleLowerCase().includes(search));
  const visible=matches.slice(0,200);q('attribute-options').innerHTML=visible.map(a=>`<option value="${esc(a.id)}">${esc(a.label)}${domain(a).exists?'':' · no values in this target'}</option>`).join('');
  q('attribute-options').value=visible.some(a=>a.id===previous)?previous:visible[0]?.id||'';
  q('attribute-search-count').textContent=`${matches.length.toLocaleString()} attributes${matches.length>200?' · showing the first 200; refine the search.':'.'}`;
  selectionDescription();
 }
 function selectionDescription(){const a=catalog.get(q('attribute-options').value);q('attribute-picker-description').textContent=a?a.description:'';root.querySelector('[data-attribute-action=choose]').disabled=!a;}
 function open(mode){pickerMode=mode;q('attribute-picker-title').textContent=mode==='color'?'Color by an attribute':'Add an attribute filter';q('attribute-search').value='';options();dialog.showModal();q('attribute-search').focus();}
 function filterMarkup(a,f){
  const d=domain(a),step=d.low===d.high||d.integer&&!(a.low===0&&a.high===1)?1:(d.high-d.low)/500;
  const numeric=a.kind==='number'?`<div class="report-range-fields"><label>Minimum${a.unit?' ('+esc(a.unit)+')':''}<input type="number" step="any" ${d.exists?'':'disabled'} data-filter-input="low" value="${f.low}" aria-label="Minimum ${esc(a.label)}"></label><label>Maximum${a.unit?' ('+esc(a.unit)+')':''}<input type="number" step="any" ${d.exists?'':'disabled'} data-filter-input="high" value="${f.high}" aria-label="Maximum ${esc(a.label)}"></label></div><div class="report-range-sliders"><label>Lower bound<input type="range" data-filter-input="low-slider" min="${d.low}" max="${d.high===d.low?d.low+1:d.high}" step="${step}" value="${f.low}" ${!d.exists||d.low===d.high?'disabled':''} aria-label="Lower bound for ${esc(a.label)}"></label><label>Upper bound<input type="range" data-filter-input="high-slider" min="${d.low}" max="${d.high===d.low?d.low+1:d.high}" step="${step}" value="${f.high}" ${!d.exists||d.low===d.high?'disabled':''} aria-label="Upper bound for ${esc(a.label)}"></label></div>`:
   `<label>Find values<input type="search" data-filter-input="category-search" placeholder="Search recorded values" aria-label="Find values for ${esc(a.label)}"></label><div class="report-category-options" data-filter-part="categories"></div><label>Text contains<input type="search" data-filter-input="text" placeholder="Any text" aria-label="Text within ${esc(a.label)}"></label>`;
  return `<div class="report-attribute-toolbar"><h3>${esc(a.label)}</h3><button type="button" class="report-button" data-attribute-action="remove" data-attribute-id="${esc(a.id)}" aria-label="Remove ${esc(a.label)} filter">Remove</button></div><p class="report-sub">${esc(a.description)}</p><div data-filter-part="histogram"></div>${numeric}<p class="report-filter-error" data-filter-part="error" aria-live="polite"></p><div class="report-filter-options"><label><input type="checkbox" data-filter-input="missing"> Include unavailable values</label>${d.multiple?'<label>For several recorded values<select data-filter-input="mode"><option value="all">Every value must match</option><option value="any">Any value can match</option></select></label>':''}</div>`;
 }
 function add(id){
  const a=catalog.get(id);if(!a)return;
  if(filters.has(id)){q('attribute-filters').querySelector(`[data-filter-id="${id}"]`).scrollIntoView({block:'nearest'});return;}
  const d=domain(a),f={low:d.low,high:d.high,includeMissing:false,mode:'all',categories:new Set(),text:'',categorySearch:''};filters.set(id,f);
  const card=document.createElement('section');card.className='report-attribute-filter';card.dataset.filterId=id;card.setAttribute('aria-label',a.label+' filter');card.innerHTML=filterMarkup(a,f);q('attribute-filters').append(card);changed();card.querySelector('input')?.focus({preventScroll:true});
 }
 function clear(){filters.clear();q('attribute-filters').replaceChildren();}
 function render(){
  q('attribute-explorer').hidden=false;
  const a=catalog.get(colorId),base=baseRows(),visible=base.filter(r=>matches(r));
  q('attribute-color-label').textContent=a?a.label:'Choose an attribute';
  q('attribute-color-description').textContent=a?(a.description+(domain(a).multiple?(a.kind==='number'?' Color uses the lowest value recorded for each ASO.':' Each recorded combination has its own color.') :'')):'';
  const focused=document.activeElement,quickFocus=focused?.dataset.attributeAction==='quick-color'?focused.dataset.attributeId:null;
  const oldHistogram=q('attribute-color-histogram').querySelector('details');if(oldHistogram)histogramOpen=oldHistogram.open;
  q('attribute-color-histogram').innerHTML=a?`<details ${histogramOpen?'open':''}><summary>Distribution · ${esc(a.label)}</summary>${histogram(a,base.filter(r=>matches(r,a.id)),filters.get(a.id))}${filters.has(a.id)?'':`<button type="button" class="report-button" data-attribute-action="filter-color">Filter this attribute</button>`}</details>`:'';
  q('attribute-color-histogram').hidden=!!a&&filters.has(a.id);
  q('attribute-quick').innerHTML=[...catalog.values()].filter(a=>a.id==='oligoai'||a.id==='effect'||a.id.startsWith('uniqueness')||a.id==='phastcons').map(a=>`<button type="button" class="report-button" data-attribute-action="quick-color" data-attribute-id="${a.id}" aria-pressed="${a.id===colorId}" title="${esc(a.label)}">${a.id==='oligoai'?'OligoAI':a.id==='effect'?'Splicing':a.id==='phastcons'?'Conservation':'Uniqueness'}</button>`).join('');
  if(quickFocus)q('attribute-quick').querySelector(`[data-attribute-id="${quickFocus}"]`)?.focus({preventScroll:true});
  q('attribute-summary').textContent=filters.size?`${visible.length.toLocaleString()} of ${base.length.toLocaleString()} ASO records match ${filters.size===1?'this filter':'all '+filters.size+' filters'}. ${filters.size>1?'Each distribution shows ASOs that pass the other filters.':'The distribution shows ASOs before this filter.'}`:'Add filters to choose ranges or categories. The plot and table show the same matching ASOs.';
  root.querySelector('[data-attribute-action=clear]').hidden=!filters.size;
  root.querySelector('[data-action=filtered-csv]').disabled=!visible.length;
  for(const [id,f] of filters){
   const a=catalog.get(id),card=q('attribute-filters').querySelector(`[data-filter-id="${id}"]`),population=base.filter(r=>matches(r,id));
   card.querySelector('[data-filter-part=histogram]').innerHTML=histogram(a,population,f);
   if(a.kind==='category'){
    const focusedCategory=card.contains(focused)?focused?.dataset.filterCategory:undefined,scroll=card.querySelector('[data-filter-part=categories]').scrollTop;
    if(!categoryChoices.has(id)){const choices=new Set();scopeRows().forEach(r=>known(a,r).forEach(v=>choices.add(String(v))));categoryChoices.set(id,choices);}
    const counts=new Map([...categoryChoices.get(id)].map(v=>[v,0]));population.forEach(r=>[...new Set(known(a,r).map(String))].forEach(v=>counts.set(v,(counts.get(v)||0)+1)));
    f.categories.forEach(v=>{if(!counts.has(v))counts.set(v,0);});
    const entries=[...counts].filter(([v])=>label(a,v).toLocaleLowerCase().includes(f.categorySearch.toLocaleLowerCase())).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,60);
    card.querySelector('[data-filter-part=categories]').innerHTML=entries.map(([v,count])=>`<label><input type="checkbox" data-filter-category="${esc(v)}" ${f.categories.has(v)?'checked':''}>${esc(label(a,v))}<span>${count.toLocaleString()}</span></label>`).join('')+(counts.size>60?'<p class="report-sub">Search to find more values.</p>':'');
    card.querySelector('[data-filter-part=categories]').scrollTop=scroll;
    if(focusedCategory!==undefined)[...card.querySelectorAll('[data-filter-category]')].find(input=>input.dataset.filterCategory===focusedCategory)?.focus({preventScroll:true});
   }
  }
 }
 function refresh(){
  domains.clear();categoryColors.clear();categoryChoices.clear();knownValues.clear();
  if(!colorId){const preferred=['oligoai','effect','uniqueness','phastcons','gc'];colorId=preferred.find(id=>catalog.has(id)&&domain(catalog.get(id)).exists)||catalog.keys().next().value||'';}
  for(const [id,f] of filters){
   const a=catalog.get(id);if(a.kind!=='number')continue;const d=domain(a),card=q('attribute-filters').querySelector(`[data-filter-id="${id}"]`);
   card.querySelectorAll('input[type=number]').forEach(input=>input.disabled=!d.exists);
   card.querySelectorAll('input[type=range]').forEach(input=>{input.min=d.low;input.max=d.high===d.low?d.low+1:d.high;input.step=d.low===d.high||d.integer&&!(a.low===0&&a.high===1)?1:(d.high-d.low)/500;input.disabled=!d.exists||d.low===d.high;input.value=input.dataset.filterInput.startsWith('low')?f.low:f.high;});
  }
 }
 function legend(){
  const a=catalog.get(colorId);if(!a)return {text:'Grey: attribute unavailable.',gradient:false};
  const d=domain(a),shown=baseRows().filter(r=>matches(r)),missing=shown.filter(r=>!known(a,r).length).length;
  if(a.kind==='number')return {text:d.exists?`${format(d.low,a.unit)} → ${format(d.high,a.unit)} · ${d.low<0&&d.high>0?'Ochre: negative. Teal: positive.':a.direction==='lower'?'Lower values are darker.':'Higher values are darker.'} Grey: unavailable (${missing.toLocaleString()}).`:`${a.label}: no recorded values for this target.`,gradient:d.exists,
   background:d.low<0&&d.high>0?`linear-gradient(90deg,#a75e2a,#eaf4f6 ${-d.low/(d.high-d.low)*100}%,#12687f)`:a.direction==='lower'?'linear-gradient(90deg,#12687f,#dbedf4)':'linear-gradient(90deg,#dbedf4,#12687f)'};
  const counts=new Map();shown.forEach(r=>{if(known(a,r).length){const k=categoryKey(a,r);counts.set(k,(counts.get(k)||0)+1);}});
  const entries=[...counts].sort((a,b)=>b[1]-a[1]).slice(0,8);
  return {text:`${counts.size.toLocaleString()} recorded categories · Grey: unavailable (${missing.toLocaleString()}).`,gradient:false,
   items:entries.map(([key,count])=>({label:categoryLabel(a,key),color:categoryPalette(a).get(key),count}))};
 }
 function describe(row){
  const ids=[...new Set([colorId,...filters.keys()])];
  return ids.map(id=>{const a=catalog.get(id),v=known(a,row);return a.label+': '+(v.length?(a.kind==='number'?v.map(x=>format(x,a.unit)).slice(0,5).join(' / '):categoryLabel(a,categoryKey(a,row))):'Unavailable');}).join(' · ');
 }
 root.addEventListener('click',event=>{
  const button=event.target.closest('[data-attribute-action]');if(!button)return;
  const action=button.dataset.attributeAction,id=button.dataset.attributeId;
  if(action==='add'||action==='color')open(action);
  if(action==='close')dialog.close();
  if(action==='choose'){const id=q('attribute-options').value;dialog.close();if(pickerMode==='color'){colorId=id;changed();}else add(id);}
  if(action==='quick-color'){colorId=id;changed();}
  if(action==='filter-color')add(colorId);
  if(action==='clear'){clear();changed();}
  if(action==='remove'){filters.delete(id);q('attribute-filters').querySelector(`[data-filter-id="${id}"]`).remove();changed();}
 });
 root.addEventListener('input',event=>{
  const input=event.target,card=input.closest('[data-filter-id]');if(!card)return;
  const id=card.dataset.filterId,f=filters.get(id),a=catalog.get(id),key=input.dataset.filterInput;
  if(input.dataset.filterCategory!==undefined){input.checked?f.categories.add(input.dataset.filterCategory):f.categories.delete(input.dataset.filterCategory);changed();return;}
  if(!key)return;
  if(key==='missing')f.includeMissing=input.checked;
  else if(key==='mode')f.mode=input.value;
  else if(key==='text')f.text=input.value;
  else if(key==='category-search')f.categorySearch=input.value;
  else {
   const name=key.startsWith('low')?'low':'high',value=Number(input.value);
   if(!input.value||!Number.isFinite(value)){card.querySelector('[data-filter-part=error]').textContent='Enter a finite number.';return;}
   const other=name==='low'?'high':'low';
   if(key.endsWith('slider')){f[name]=value;if(f.low>f.high)f[other]=value;}
   else {const low=name==='low'?value:f.low,high=name==='high'?value:f.high;if(low>high){card.querySelector('[data-filter-part=error]').textContent='Minimum must be at most maximum.';return;}f[name]=value;}
   card.querySelector('[data-filter-part=error]').textContent='';
   for(const bound of ['low','high']){card.querySelector(`[data-filter-input=${bound}]`).value=f[bound];card.querySelector(`[data-filter-input=${bound}-slider]`).value=f[bound];}
  }
  changed();
 });
 q('attribute-search').addEventListener('input',options);q('attribute-options').addEventListener('change',selectionDescription);
 q('attribute-options').addEventListener('dblclick',()=>root.querySelector('[data-attribute-action=choose]').click());
 dialog.addEventListener('keydown',event=>{if(event.key==='Enter'&&['INPUT','SELECT'].includes(event.target.tagName)){event.preventDefault();root.querySelector('[data-attribute-action=choose]').click();}});
 const exportAttributes=()=>[...new Set([colorId,...filters.keys()])].map(id=>catalog.get(id)).filter(Boolean);
 return {matches,render,refresh,clear,color,describe,legend,values,known,
  tableAttributes:()=>exportAttributes().filter(a=>!['oligoai','effect','gc','native_rank'].includes(a.id)),
  sortValue:(id,row)=>{const a=catalog.get(id);return !a||!known(a,row).length?null:a.kind==='number'?numeric(a,row):categoryLabel(a,categoryKey(a,row));},
  cell:(a,row)=>{const v=known(a,row),missing=values(a,row).length-v.length;return v.length?(a.kind==='number'?v.slice(0,5).map(x=>format(x,a.unit)).join(' / ')+(v.length>5?' · '+(v.length-5)+' more values':''):categoryLabel(a,categoryKey(a,row)))+(missing?' · '+missing+' unavailable':''):'Unavailable';},
  exportValue:(a,row)=>{const v=values(a,row);return v.length===1&&v[0]==null?'':v.map(x=>x==null?'Unavailable':Object.hasOwn(a.labels||{},String(x))?a.labels[String(x)]:x).join('; ');},
  metric:()=>{const a=catalog.get(colorId);return a?{...a,...domain(a),value:row=>a.kind==='number'?numeric(a,row):null,format:value=>format(value,a.unit),
   tick:value=>finite(value)?value.toLocaleString(undefined,{maximumSignificantDigits:4,notation:Math.abs(value)>=10000?'compact':'standard'}):'Unavailable'}:null;},
  exportAttributes};
}
'''
