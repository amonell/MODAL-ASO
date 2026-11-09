"""A single-file reading copy with bounded, explicitly linked downloads.

Scientific text, tables and embedded report records are not recomputed. The
working run remains the source for large evidence, references and linked reports.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
from pathlib import Path
import re

from .report_exports import REPORT_SUFFIXES, _HTMLLinks, _reference, asset_exclusion

MAX_DOWNLOAD_BYTES = 512 * 1024
MAX_DOWNLOAD_TOTAL_BYTES = 1024 * 1024
MARKER = 'id="modal-portable-assets"'

_SCRIPT = r'''<script>
(function(){
  const packet=JSON.parse(document.getElementById('modal-portable-assets').textContent);
  const urls=new Map();
  const normalize=(value)=>{try{return new URL(value,'https://report.invalid/').pathname;}catch(e){return value;}};
  const aliases=new Map();
  for(const [id,item] of Object.entries(packet.files)){
    for(const alias of item.aliases) aliases.set(normalize(alias),id);
  }
  function download(id){
    if(!urls.has(id)){
      const item=packet.files[id];
      const bytes=Uint8Array.from(atob(item.base64),c=>c.charCodeAt(0));
      urls.set(id,URL.createObjectURL(new Blob([bytes],{type:item.mime})));
    }
    return urls.get(id);
  }
  function rewrite(a){
    const href=a.getAttribute('href')||'';
    if(href.startsWith('#portable-file-')){
      const id=href.slice('#portable-file-'.length);
      if(packet.files[id]){a.href=download(id);a.download=packet.files[id].name;}
      return;
    }
    if(!href||href.startsWith('#')||/^(https?:|mailto:|data:|blob:|tel:)/i.test(href))return;
    const id=aliases.get(normalize(href));
    if(id){a.href=download(id);a.download=packet.files[id].name;}
    else{
      a.setAttribute('href','#portable-other-files');
      a.title='View this file’s location and availability.';
    }
  }
  function scan(node){
    if(node.nodeType!==1)return;
    if(node.matches('a[href]'))rewrite(node);
    node.querySelectorAll('a[href]').forEach(rewrite);
  }
  scan(document.body);
  new MutationObserver(records=>records.forEach(record=>{
    if(record.type==='attributes')rewrite(record.target);
    else record.addedNodes.forEach(scan);
  })).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['href']});
})();
</script>'''


def standalone_html(document: str, *, source_path: Path, source_root: Path):
    """Return self-contained HTML and its size/scope receipt, without model calls."""
    if MARKER in document:
        match = re.search(r'<script id="modal-portable-assets" type="application/json">(.*?)</script>',document,re.S)
        packet = json.loads(match[1])
        return document, {'embedded_file_count':len(packet['files']),
                          'embedded_bytes':sum(f['bytes'] for f in packet['files'].values()),
                          'omitted_links':packet['omitted']}
    source, root = Path(source_path).resolve(), Path(source_root).resolve()
    if not source.is_relative_to(root):
        raise ValueError('standalone_report_source_outside_run:'+str(source))
    files, admitted, omitted, replacements, by_content = {}, {}, [], [], {}
    total = 0
    links = _HTMLLinks(document)
    # Reserve the bounded download budget for the reader's main output files.
    # Detailed evidence links can appear earlier in the document.
    ordered_links = sorted(links.links, key=lambda link: (link[0] not in links.primary_downloads, link[0]))
    for start, end, value, _ in ordered_links:
        ref = _reference(source, root, value)
        if ref is None:
            continue
        path, reason = ref.get('path'), ref.get('reason')
        if path == source:
            replacements.append((start,end,ref.get('suffix','') or '#'))
            continue
        if path and not reason:
            if path.suffix.lower() in REPORT_SUFFIXES:
                reason = 'separate_report'
            else:
                reason = asset_exclusion(path,max_bytes=MAX_DOWNLOAD_BYTES)
            if not reason and path not in admitted:
                size = path.stat().st_size
                data = path.read_bytes()
                if len(data)!=size:
                    raise ValueError('standalone_report_source_changed:'+str(path))
                digest = hashlib.sha256(data).hexdigest()
                if digest in by_content:
                    admitted[path] = by_content[digest]
                    files[admitted[path]]['aliases'].extend([path.relative_to(root).as_posix(),str(path),path.as_uri()])
                elif total+size > MAX_DOWNLOAD_TOTAL_BYTES:
                    reason = 'download_size_limit'
                else:
                    key = digest[:16]
                    mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
                    files[key] = {'name':path.name,'mime':mime,'bytes':size,
                        'sha256':digest,
                        'base64':base64.b64encode(data).decode('ascii'),
                        'aliases':[path.relative_to(root).as_posix(),str(path),path.as_uri()]}
                    admitted[path]=key
                    by_content[digest]=key
                    total += size
        if reason:
            omitted.append({'name':path.name if path else Path(value).name,'reason':reason,
                            'bytes':path.stat().st_size if path and path.is_file() else None})
            # hrefs point to an explanation; resource attributes get no network
            # or filesystem fallback. Native reports inline scripts and styles.
            prefix=document[max(0,start-12):start]
            target='#portable-other-files' if re.search(r'href\s*=\s*["\']?$',prefix,re.I) else 'data:,'
        else:
            key=admitted[path]
            files[key]['aliases'].append(value.split('#',1)[0].split('?',1)[0])
            prefix=document[max(0,start-12):start]
            if re.search(r'href\s*=\s*["\']?$',prefix,re.I):
                target='#portable-file-'+key
            else:
                target='data:'+files[key]['mime']+';base64,'+files[key]['base64']
        replacements.append((start,end,target))
    for start,end,target in sorted(replacements, reverse=True):
        document=document[:start]+html.escape(target,quote=True)+document[end:]
    packet={'files':files,'omitted':omitted}
    payload=json.dumps(packet,separators=(',',':')).replace('<','\\u003c').replace('&','\\u0026')
    download_links=''.join('<li id="portable-file-'+key+'"><a href="#portable-file-'+key+'">'
        +html.escape(item['name'])+'</a> ('+str(round(item['bytes']/1024,1))+' KB)</li>' for key,item in files.items())
    reason_labels = {'reference_or_runtime_asset': 'Reference or model file',
                     'large_evidence_file': 'Large file', 'download_size_limit': 'Download size limit',
                     'separate_report': 'Separate report', 'file_unavailable': 'File unavailable',
                     'outside_producing_run': 'Outside this run', 'invalid_local_reference': 'Location unavailable'}
    other_files = sorted({(item['name'], reason_labels.get(item['reason'], item['reason'])) for item in omitted})
    section=('<section id="portable-files" style="max-width:1120px;margin:24px auto;padding:0 24px;overflow-wrap:anywhere">'
        '<details id="portable-other-files"><summary>Files and download details</summary>'
        + ('<h3>Included downloads</h3><ul>'+download_links+'</ul>' if files else '')
        + '<noscript>Enable JavaScript to save the embedded downloads. The report text and tables remain readable.</noscript>'
        + ('<h3>Additional files</h3><p>Available files listed here can be opened from the original run folder.</p><ul>'
           + ''.join('<li>'+html.escape(name)+' — '+html.escape(reason)+'</li>' for name, reason in other_files)+'</ul>' if other_files else '')
        + '</details></section>'
        '<script id="modal-portable-assets" type="application/json">'+payload+'</script>'+_SCRIPT)
    document=document.replace('</body>',section+'</body>') if '</body>' in document else document+section
    return document, {'embedded_file_count':len(files),'embedded_bytes':total,'omitted_links':omitted}


def export_standalone_html(source, destination, *, source_root):
    source, destination = Path(source), Path(destination)
    if source.resolve()==destination.resolve():
        raise ValueError('standalone_report_would_replace_source:'+str(source))
    data=source.read_bytes()
    # Combined reports need their child documents. Keep those as a bounded
    # portable tree so iframe navigation and each child's relative links work.
    if re.search(rb'<iframe\b[^>]*\bsrc\s*=', data, re.I):
        from .report_exports import export_native_report
        return export_native_report(source, destination, source_root=source_root)
    document,receipt=standalone_html(data.decode('utf-8'),source_path=source,source_root=source_root)
    output=document.encode('utf-8')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(output)
    return {'source':str(source),'destination':str(destination),
        'source_sha256':hashlib.sha256(data).hexdigest(),'sha256':hashlib.sha256(output).hexdigest(),
        'method':'standalone_html','bytes':len(output),**receipt}
