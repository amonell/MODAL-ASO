"""Lossless one-row-per-molecule CSV projection for the offline report."""

SELECTION_EXPORT_SCRIPT = r'''
function selectedMoleculeCsv(selected,byMolecule,contexts,attributeView,evidenceFilename='ASO_selection_contexts.json'){
 const fields=['name','antisense_5to3','ordering_notation','chemistry','binding_intervals_1based','strand','design_group','program_id','job_id','response_site','native_rank','display_order','predicted_splice_effect','deletion_prediction','replacement_prediction','gc_fraction','off_target_exact','off_target_1mm','off_target_2mm','off_target_mismatch_budget','off_target_unit','off_target_status','splice_checks','panel_order','selection_note','panel_role','recommendation_tradeoff'];
 const activityFields=['value','transcript_id','species','context_basis','location','dosage_nm','transfection_method','dose_basis','method_basis','unmodeled_conditions','unresolved_conditions','status','reason','model_sha256','source_revision','input_id'];
 const attributes=attributeView.exportAttributes(),allAttributes=attributeView.allAttributes?.()||attributes,junctions=new Map();
 const detailAttributes=allAttributes.filter(a=>a.group==='Common attributes'||attributes.some(p=>p.id===a.id));
 const groups=selected.map(r=>byMolecule.get(r.molecule)||[r]);
 groups.forEach(group=>group.forEach(r=>(r.junction_measurements||[]).forEach(j=>junctions.set(j.id,j))));
 const junctionList=[...junctions.values()],wideJunctions=junctionList.length<=32?junctionList:[];
 const labels=new Map();wideJunctions.forEach(j=>labels.set(j.label,(labels.get(j.label)||0)+1));
 const junctionLabel=j=>j.label+' predicted splice effect'+(labels.get(j.label)>1?' · '+[j.scope.reference,j.scope.job,j.scope.chrom,j.scope.position,j.scope.model].filter(v=>v!=null).join(' · ')+' · '+j.id.slice(-6):'');
 const headers=[...fields,...activityFields.map(k=>'oligoai_'+(k==='value'?'predicted_rna_reduction_percent':k)),...attributes.map(a=>'Attribute: '+a.label),...wideJunctions.map(junctionLabel),'selection_order','context_count','full_context_evidence_file','context_evidence_json'];
 const shared=values=>values.length&&values.every(v=>JSON.stringify(v??null)===JSON.stringify(values[0]??null))?values[0]??'':'';
 const base=r=>[r.name,r.aso,r.chemistry.notation,r.chemistry.label+'; '+r.chemistry.detail,r.location,r.strand,contexts.get(r.context)?.label,r.origin.program,r.origin.job,r.origin.route,r.native_rank,r.order,r.effect,...[0,1].map(i=>r.predictions[i]??null),r.gc,...r.off_targets,r.off_target_budget,r.off_target_unit,r.off_target_status,r.guard.summary,r.panel_rank,r.selection_note,r.recommendation?.panel_role,r.recommendation?.tradeoff];
 const values=groups.map((group,i)=>{
  const cells=group.map(base),activities=group.every(r=>r.oligoai?.length)?group.flatMap(r=>r.oligoai):[];
  const evidence=group.map(r=>({...r,context:Object.fromEntries(['id','label','transcript','program','job','direction'].map(k=>[k,(contexts.get(r.context)||{id:r.context})[k]])),attributes:Object.fromEntries(detailAttributes.map(a=>[a.id,{label:a.label,unit:a.unit||'',values:attributeView.values(a,r)}]))}));
  return [...fields.map((_,index)=>shared(cells.map(c=>c[index]))),...activityFields.map(k=>{
    const value=shared(activities.map(a=>a[k]));return value&&typeof value==='object'?JSON.stringify(value):value;
   }),...attributes.map(a=>shared(group.map(r=>attributeView.exportValue(a,r)))),
   ...wideJunctions.map(j=>shared(group.flatMap(r=>(r.junction_measurements||[]).filter(m=>m.id===j.id).map(m=>m.mean)))),
   i+1,group.length,evidenceFilename,JSON.stringify(evidence)];
 });
 return [headers,...values];
}
function selectedMoleculeEvidence(selected,byMolecule,contexts,attributeView){
 const attributes=attributeView.allAttributes?.()||attributeView.exportAttributes();
 const records=selected.flatMap(r=>byMolecule.get(r.molecule)||[r]);
 return {format:'modal-aso-context-evidence-v1',
  molecules:selected.map((r,i)=>({molecule:r.molecule,selection_order:i+1,record_ids:(byMolecule.get(r.molecule)||[r]).map(c=>c.id)})),
  contexts:[...new Set(records.map(r=>r.context))].map(id=>contexts.get(id)||{id}),records,
  attributes:attributes.map(a=>({id:a.id,label:a.label,unit:a.unit||'',description:a.description,
    kind:a.kind,labels:a.labels,values:records.map(r=>attributeView.values(a,r))}))};
}
'''
