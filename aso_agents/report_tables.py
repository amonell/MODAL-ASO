"""Bound the initial table DOM while retaining every exported result row."""
from __future__ import annotations

import base64
import csv
import hashlib
from html.parser import HTMLParser
import io
import json


PAGE_SIZE = 25


class _Cells(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.cells = []
        self.anchors = []
        self.search_parts = []
        self.parts = None
        self.recorded_depth = 0
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.anchors.append(attrs['id'])
        if tag in {'td', 'th'}:
            self.parts = []
        if self.parts is None:
            return
        if self.recorded_depth:
            self.recorded_depth += 1
        elif 'data-recorded-value' in attrs:
            self.parts.append(attrs['data-recorded-value'])
            self.search_parts.append(attrs['data-recorded-value'])
            self.recorded_depth = 1
        elif tag == 'br':
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if self.recorded_depth:
            self.recorded_depth -= 1
        if tag in {'td', 'th'} and self.parts is not None:
            self.cells.append(''.join(self.parts).strip())
            self.parts = None

    def handle_data(self, data):
        if self.parts is not None:
            self.search_parts.append(data)
        if self.parts is not None and not self.recorded_depth:
            self.parts.append(data)


class _Tables(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.offsets = [0]
        for line in source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.depth = 0
        self.current = None
        self.tables = []
        self.feed(source)

    def source_offset(self):
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.depth += 1
            if self.depth == 1:
                self.current = {'start': self.source_offset(), 'rows': [], 'bodies': 0, 'safe': True}
            else:
                self.current['safe'] = False
        if self.depth != 1:
            return
        table = self.current
        if tag == 'tbody':
            table['bodies'] += 1
            table['body_start'] = self.source_offset() + len(self.get_starttag_text())
            table['in_body'] = True
        elif tag == 'tr' and table.get('in_body'):
            table['row_start'] = self.source_offset()
        elif tag in {'td', 'th'} and any(key in {'rowspan', 'colspan'} for key, _ in attrs):
            table['safe'] = False

    def handle_endtag(self, tag):
        if self.depth == 1:
            table = self.current
            end = self.source.find('>', self.source_offset()) + 1
            if tag == 'tr' and table.get('in_body') and 'row_start' in table:
                table['rows'].append(self.source[table.pop('row_start'):end])
            elif tag == 'tbody':
                table['body_end'] = self.source_offset()
                table['in_body'] = False
            elif tag == 'table':
                table['end'] = end
                if table['safe'] and table['bodies'] == 1 and len(table['rows']) > PAGE_SIZE:
                    self.tables.append(table)
                self.current = None
        if tag == 'table':
            self.depth -= 1


def page_large_tables(fragment):
    """Keep original order and markup; search and paging are presentation only.

    Complex merged or nested tables retain their existing presentation. Without
    JavaScript, the first page and a complete, embedded CSV remain available.
    """
    parser = _Tables(fragment)
    for index, table in reversed(list(enumerate(parser.tables))):
        start, end = table['start'], table['end']
        rows = table['rows']
        header = fragment[start:table['body_start']]
        footer = fragment[table['body_end']:end]
        identifier = 'reader-table-' + hashlib.sha256(fragment[start:end].encode()).hexdigest()[:12] + '-' + str(index)
        cells = [_Cells(row) for row in rows]
        output = io.StringIO(newline='')
        writer = csv.writer(output)
        writer.writerow(_Cells(header).cells)
        writer.writerows(row.cells for row in cells)
        csv_data = base64.b64encode(output.getvalue().encode('utf-8-sig')).decode('ascii')
        records = [{'html': row, 'search': ' '.join(cell.search_parts), 'anchors': cell.anchors}
                   for row, cell in zip(rows, cells)]
        data = json.dumps(records, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
        total = len(rows)
        replacement = (
            f'<div class="reader-paged-table" id="{identifier}" data-row-count="{total}">'
            '<div class="reader-table-tools">'
            f'<p class="reader-table-status" aria-live="polite">Rows 1–{PAGE_SIZE} of {total:,}</p>'
            f'<a class="reader-table-download" download="aso-results-{index + 1}.csv" '
            f'href="data:text/csv;base64,{csv_data}">Download all {total:,} rows (CSV)</a>'
            '<div class="reader-js-table-controls" hidden>'
            f'<label for="{identifier}-search">Search this table</label>'
            f'<input id="{identifier}-search" class="reader-table-search" type="search" '
            'placeholder="Candidate, sequence or result" autocomplete="off">'
            f'<label for="{identifier}-size">Rows per page</label>'
            f'<select id="{identifier}-size" class="reader-table-size">'
            '<option>25</option><option>50</option><option>100</option></select>'
            '<button type="button" class="reader-table-previous" disabled>Previous</button>'
            '<button type="button" class="reader-table-next">Next</button>'
            '</div></div>'
            + header + ''.join(rows[:PAGE_SIZE]) + footer
            + '<noscript><p>Enable JavaScript to search and browse every row, or download the complete table above.</p></noscript>'
            + '<script type="application/json" class="reader-table-data">' + data + '</script></div>'
        )
        fragment = fragment[:start] + replacement + fragment[end:]
    return fragment


TABLE_CSS = """
.reader-table-tools{position:sticky;left:0;padding:12px;display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;background:var(--reader-paper,#fff);max-width:100%;box-sizing:border-box}
.reader-table-tools p{margin:0}.reader-table-download{font-size:.9rem}
.reader-js-table-controls:not([hidden]){display:flex;align-items:center;gap:8px;flex-wrap:wrap;width:100%}
.reader-table-tools input{min-width:0;flex:1 1 15rem;max-width:28rem}
.reader-table-tools input,.reader-table-tools select,.reader-table-tools button{font:inherit;padding:7px 10px;border:1px solid var(--reader-line,#dbe4de);border-radius:6px;background:#fff;color:inherit}
.reader-table-tools button{cursor:pointer}.reader-table-tools button:disabled{opacity:.45;cursor:default}
@media(max-width:600px){.reader-table-tools input{flex-basis:100%;max-width:100%}.reader-table-tools{gap:8px;padding:10px}}
@media print{.reader-table-tools{position:static}.reader-js-table-controls,.reader-table-download{display:none!important}}
"""


TABLE_SCRIPT = """<script>
(() => {
  const controllers=[];
  document.querySelectorAll('.reader-paged-table').forEach(root=>{
    const table=root.querySelector('table'), body=table.tBodies[0];
    const input=root.querySelector('.reader-table-search'), size=root.querySelector('.reader-table-size');
    const previous=root.querySelector('.reader-table-previous'), next=root.querySelector('.reader-table-next');
    const status=root.querySelector('.reader-table-status');
    const total=Number(root.dataset.rowCount);
    let records=null, matches=null, page=0, timer;
    const load=()=>{if(!records){records=JSON.parse(root.querySelector('.reader-table-data').textContent);matches=records;}return records;};
    function draw(printing=false){
      load();
      const step=Number(size.value), start=page*step;
      const visible=printing?matches:matches.slice(start,start+step);
      body.innerHTML=visible.map(row=>row.html).join('');
      const columns=[...table.querySelectorAll('thead th')].map((th,i)=>th.classList.contains('reader-detail-column')?i:-1).filter(i=>i>=0);
      [...body.rows].forEach(row=>columns.forEach(i=>row.cells[i]?.classList.add('reader-detail-column')));
      const suffix=matches.length===total?`of ${total.toLocaleString()}`:`of ${matches.length.toLocaleString()} matches · ${total.toLocaleString()} rows total`;
      status.textContent=!matches.length?`No matching rows · ${total.toLocaleString()} rows total`:
        `Rows ${(printing?1:start+1).toLocaleString()}–${(printing?matches.length:Math.min(start+step,matches.length)).toLocaleString()} ${suffix}`;
      previous.disabled=page===0;next.disabled=(page+1)*step>=matches.length;
    }
    function search(){
      const query=input.value.trim().toLocaleLowerCase();
      matches=load().filter(row=>!query||row.search.toLocaleLowerCase().includes(query));page=0;draw();
    }
    input.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(search,120);});
    size.addEventListener('change',()=>{clearTimeout(timer);search();});
    previous.addEventListener('click',()=>{if(page>0){page--;draw();}});
    next.addEventListener('click',()=>{load();if((page+1)*Number(size.value)<matches.length){page++;draw();}});
    root.querySelector('.reader-js-table-controls').hidden=false;
    controllers.push({draw,reveal(id){const index=load().findIndex(row=>row.anchors.includes(id));
      if(index<0)return false;clearTimeout(timer);input.value='';matches=records;page=Math.floor(index/Number(size.value));draw();return true;}});
  });
  window.revealReaderTableAnchor=id=>controllers.some(controller=>controller.reveal(id));
  addEventListener('beforeprint',()=>controllers.forEach(controller=>controller.draw(true)));
  addEventListener('afterprint',()=>controllers.forEach(controller=>controller.draw()));
})();
</script>"""
