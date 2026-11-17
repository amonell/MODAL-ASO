"""Geometry for readable offline binding maps, independent of the design type."""

BINDING_MAP_SCRIPT = r'''
function bindingRegions(rows){
 // Recorded safeguards can be far from any ASO. Only binding intervals set
 // the map extent; the complete site catalog remains available separately.
 const intervals=rows.flatMap(r=>r.intervals.map(p=>[Math.max(1,p.start-12),p.end+12])).sort((a,b)=>a[0]-b[0]);
 const regions=[];
 for(const interval of intervals){const last=regions.at(-1);
  if(last&&interval[0]<=last[1]+70)last[1]=Math.max(last[1],interval[1]);else regions.push(interval.slice());
 }
 return regions;
}
function bindingAxes(regions,width,left,right,gap=36){
 gap=Math.min(gap,(width-left-right)/Math.max(1,regions.length)/4);
 const total=regions.reduce((sum,[a,b])=>sum+b-a+1,0),available=width-left-right-gap*(regions.length-1);
 let offset=left;
 return regions.map(([a,b])=>{const size=(b-a+1)/total*available,axis={a,b,x:offset,width:size};offset+=size+gap;return axis;});
}
function bindingLabels(sites,measure,left,right,maxRows=3){
 const lanes=Array.from({length:maxRows},()=>[]),placed=[];
 for(const site of [...sites].sort((a,b)=>Number(b.target)-Number(a.target)||a.x-b.x)){
  const full=(site.target?'Target · ':'')+site.label,limit=Math.min(280,right-left);
  let label=full;
  if(measure(label)>limit){while(label.length&&measure(label+'…')>limit)label=label.slice(0,-1);label+='…';}
  const width=measure(label),start=Math.max(left,Math.min(right-width,site.x-width/2));
  const lane=lanes.findIndex(items=>items.every(item=>start>item.end+12||start+width+12<item.start));
  if(lane<0)continue;
  lanes[lane].push({start,end:start+width});placed.push({...site,text:label,left:start,width,lane});
 }
 return placed;
}
function bindingTicks(axes,measure){
 // Prioritize both ends of each region, then add interior ticks only where
 // their measured labels fit. Coordinates stay exact at every scale.
 const ticks=[];
 for(const axis of axes){
  const proposals=[{value:axis.a,x:axis.x,align:'left'},{value:axis.b,x:axis.x+axis.width,align:'right'}];
  for(const fraction of [.5,.25,.75]){const value=Math.round(axis.a+(axis.b-axis.a)*fraction);
   proposals.push({value,x:axis.x+(value-axis.a)/(axis.b-axis.a+1)*axis.width,align:'center'});
  }
  for(const tick of proposals){const text=tick.value.toLocaleString(),width=measure(text),left=tick.x-(tick.align==='right'?width:tick.align==='center'?width/2:0);
   if(left<axis.x-.5||left+width>axis.x+axis.width+.5||ticks.some(t=>left<t.right+16&&left+width+16>t.left))continue;
   ticks.push({...tick,text,left,right:left+width});
  }
 }
 return ticks;
}
'''
