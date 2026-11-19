"""Portable, paginated display of recorded candidate exclusions."""
import html
import json
from urllib.parse import quote


def filter_history_html(data):
    if not data or not data.get('records'):
        return ''
    rows = data['records']
    esc = lambda v: html.escape(str(v if v is not None else ''), quote=True)
    fields = [('aso', 'ASO · 5′→3′'), ('location', 'Binding location'),
              ('gc_display', 'GC'), ('status', 'Disposition'), ('stage', 'Stage'), ('reason', 'Reason')]
    def render_row(row):
        cells = ''.join('<td>' + esc(row.get(key) or 'Unavailable') + '</td>' for key, _ in fields)
        details = '<details><summary>Source &amp; chemistry</summary><p>' + esc(row.get('group')) + '</p><p>' + esc(row.get('chemistry') or 'Chemistry unavailable') + '</p>'
        if row.get('comparison_sequence'):
            details += '<p>Recorded comparison sequence: ' + esc(row['comparison_sequence']) + '</p>'
        if row.get('notation'):
            details += '<p class="filter-notation">' + esc(row['notation']) + '</p>'
        details += ''.join('<p><a href="' + esc(quote(path, safe='/')) + '">' + esc(path) + '</a></p>' for path in row.get('sources') or [])
        return '<tr>' + cells + '<td>' + details + '</details></td></tr>'
    filtered = sum(r['status'] == 'Filtered out' for r in rows)
    unknown = sum(r['status'] == 'Removal reason unavailable' for r in rows)
    payload = json.dumps({'records': rows}, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')
    return '''<section class="filter-history reader-section" id="filtered-candidates" tabindex="-1" aria-label="Filtered candidates">
<h2>Filtered candidates</h2><p class="filter-intro">Candidates excluded by a recorded filter, or absent from a later recorded stage. Scored candidates that were not recommended remain in the main ASO table.</p>
<div class="filter-facts"><div><strong>''' + f'{len(rows):,}' + '''</strong><span>candidate records</span></div><div><strong>''' + f'{filtered:,}' + '''</strong><span>recorded exclusions</span></div><div><strong>''' + f'{unknown:,}' + '''</strong><span>reason unavailable</span></div></div>
''' + ('<p class="filter-note">' + esc(data['note']) + '</p>' if data.get('note') else '') + '''
<p class="filter-sub">Binding locations use 1-based inclusive genomic coordinates. Missing sequences or locations are shown as unavailable. Inclusion here does not recommend a candidate for ordering.</p>
<div class="filter-controls"><label>Sequence, position or reason<input type="search" data-filter="search" placeholder="Search filtered candidates…"></label><label>Disposition<select data-filter="status"><option value="">All dispositions</option><option>Filtered out</option><option>Removal reason unavailable</option></select></label><label>Stage<select data-filter="stage"><option value="">All stages</option>''' + ''.join('<option>' + esc(s) + '</option>' for s in sorted({r['stage'] for r in rows})) + '''</select></label><button type="button" data-filter="csv">Download filtered CSV</button></div>
<div class="filter-table-wrap" tabindex="0" role="region" aria-label="Filtered candidate table"><table data-report-native="true"><thead><tr>''' + ''.join('<th scope="col">' + esc(label) + '</th>' for _, label in fields) + '''<th scope="col">Evidence</th></tr></thead><tbody>''' + ''.join(render_row(r) for r in rows[:25]) + '''</tbody></table></div>
<div class="filter-pagination"><label>Rows per page <select data-filter="size"><option>25</option><option>50</option><option>100</option></select></label><span data-filter="count" aria-live="polite">''' + f'{len(rows):,} records' + '''</span><div><button type="button" data-filter="previous">Previous</button><button type="button" data-filter="next">Next</button></div></div>
<noscript><p>The first 25 records are shown. Enable JavaScript to search, page through all candidates and download CSV.</p></noscript>
<script type="application/json" class="filter-history-data">''' + payload + '''</script></section>''' + FILTER_CSS + FILTER_SCRIPT


FILTER_CSS = '''<style>
.filter-history{margin:32px 0;padding:24px;border:1px solid #d6e2e6;border-radius:12px;background:#fff;scroll-margin-top:100px;min-width:0;color:#203844}
.filter-history h2{margin:0 0 12px}.filter-intro,.filter-sub{line-height:1.6}.filter-sub{font-size:13px;color:#536975}
.filter-facts{display:flex;flex-wrap:wrap;gap:16px;margin:20px 0}.filter-facts>div{background:#f1f6f7;border-radius:8px;padding:14px 20px;flex:1;min-width:130px}.filter-facts strong{display:block;font-size:26px;color:#194f59}.filter-facts span{font-size:13px}.filter-note{padding:16px;border-left:4px solid #a77625;background:#fcf8ee;line-height:1.6}
.filter-controls{display:flex;flex-wrap:wrap;align-items:end;gap:12px;margin:20px 0}.filter-controls label,.filter-pagination label{display:grid;gap:5px;font-size:13px}.filter-controls input,.filter-controls select,.filter-pagination select,.filter-history button{font:inherit;border:1px solid #c6d6dc;border-radius:6px;background:white;color:#203844;padding:8px 10px;max-width:100%}.filter-history button{cursor:pointer}.filter-history button:disabled{opacity:.45;cursor:default}.filter-controls>label:first-child{flex:1;min-width:180px}
.filter-table-wrap{overflow:auto;width:100%;border:1px solid #dce5e8;border-radius:8px}.filter-history table{border-collapse:collapse;width:100%;font-size:13px;min-width:850px}.filter-history th{text-align:left;background:#eef4f6;color:#254653}.filter-history td,.filter-history th{padding:12px;vertical-align:top;border-bottom:1px solid #e4ecee}.filter-history td:first-child{font-family:ui-monospace,monospace;white-space:nowrap}.filter-history td:nth-child(2){min-width:180px}.filter-history td:nth-child(3){white-space:nowrap}.filter-history td:nth-child(6){min-width:220px}.filter-history td:last-child{min-width:180px;max-width:280px;overflow-wrap:anywhere}.filter-history details{font-size:12px}.filter-history summary{cursor:pointer;color:#176370}.filter-notation{max-width:260px;overflow-wrap:anywhere}.filter-pagination{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:16px}
@media(max-width:600px){.filter-history{padding:14px}.filter-controls{align-items:stretch}.filter-controls>label{flex:1;min-width:min(140px,100%);grid-template-columns:minmax(0,1fr)}.filter-controls input,.filter-controls select{box-sizing:border-box;min-width:0;width:100%}.filter-pagination{font-size:12px}}
</style>'''


FILTER_SCRIPT = r'''<script>
(()=>{'use strict';for(const root of document.querySelectorAll('.filter-history')){
if(root.dataset.ready)continue;root.dataset.ready='true';
const rows=JSON.parse(root.querySelector('.filter-history-data').textContent).records;
const get=name=>root.querySelector('[data-filter="'+name+'"]');
const esc=v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
const fields=['aso','location','gc_display','status','stage','reason'];
const search=rows.map(r=>[...fields.map(k=>r[k]),r.group,r.chemistry,r.comparison_sequence].join(' ').toLowerCase());let page=0,matching=rows;
function draw(){const query=get('search').value.toLowerCase().trim();matching=rows.filter((r,i)=>(!query||search[i].includes(query))&&(!get('status').value||r.status===get('status').value)&&(!get('stage').value||r.stage===get('stage').value));
const size=Number(get('size').value),pages=Math.max(1,Math.ceil(matching.length/size));page=Math.min(page,pages-1);
root.querySelector('tbody').innerHTML=matching.slice(page*size,(page+1)*size).map(r=>'<tr>'+fields.map(k=>'<td>'+esc(r[k]||'Unavailable')+'</td>').join('')+'<td><details><summary>Source &amp; chemistry</summary><p>'+esc(r.group)+'</p><p>'+esc(r.chemistry||'Chemistry unavailable')+'</p>'+(r.comparison_sequence?'<p>Recorded comparison sequence: '+esc(r.comparison_sequence)+'</p>':'')+(r.notation?'<p class="filter-notation">'+esc(r.notation)+'</p>':'')+(r.sources||[]).map(p=>'<p><a href="'+esc(p.split('/').map(encodeURIComponent).join('/'))+'">'+esc(p)+'</a></p>').join('')+'</details></td></tr>').join('')||'<tr><td colspan="7">No matching candidates.</td></tr>';
get('count').textContent=matching.length?`${page*size+1}–${Math.min((page+1)*size,matching.length)} of ${matching.length.toLocaleString()} records`:'0 matching records';get('previous').disabled=page===0;get('next').disabled=page+1>=pages;get('csv').disabled=!matching.length;
}
for(const name of ['search','status','stage','size'])get(name).addEventListener(name==='search'?'input':'change',()=>{page=0;draw()});
get('previous').onclick=()=>{page--;draw()};get('next').onclick=()=>{page++;draw()};
get('csv').onclick=()=>{const columns=['aso','location','strand','gc','status','stage','reason','group','chemistry','notation','comparison_sequence','sources'];const cell=v=>{let t=String(v??'');if(typeof v==='string'&&['=','+','@','-'].includes(t.trimStart().charAt(0)))t="'"+t;return '"'+t.replaceAll('"','""')+'"'};const text=[columns,...matching.map(r=>columns.map(k=>Array.isArray(r[k])?r[k].join('; '):r[k]))].map(row=>row.map(cell).join(',')).join('\r\n')+'\r\n';const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='filtered_candidates.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};draw();
}})();</script>'''
