"""Reference identities derived from the inputs actually passed to a design job."""
import hashlib
import json


def executed_reference(job, label, entry):
    metadata = entry.get('metadata') or {}
    transcript = entry.get('transcript_id') or entry.get('id') or metadata.get('id')
    versioned = metadata.get('versioned_id') or transcript
    geometry = {'chrom': entry.get('chrom') or metadata.get('seq_region_name'),
                'strand': entry.get('strand') or metadata.get('strand'),
                **{key: entry.get(key) or metadata.get(key) for key in ('species', 'assembly_name', 'gene_id')},
                'exons_tx': [{key: exon.get(key) for key in ('index', 'start', 'end')}
                             for exon in entry.get('exons_tx') or []]}
    accession = metadata.get('reference_accession_resolution') or {}
    return {'version': 1, 'job_id': job.get('job_id'), 'program_id': job.get('program_id'),
            'isoform_label': label, 'transcript_id': transcript, 'versioned_id': versioned,
            'structure_sha256': hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest(),
            'reference_accession_resolution': accession,
            # An accession crosswalk attests identity, not full cDNA equality.
            'full_rna_sequence_equivalence': 'not_established_by_this_receipt'}


def merged_executed_references(jobs, results):
    """Do not assign a global template to a multi-reference or incomplete result."""
    records = [results.get(job.get('job_id'), {}).get('executed_reference') for job in jobs
               if not results.get(job.get('job_id'), {}).get('empty_by_constraints')]
    resolved = [record for record in records if isinstance(record, dict) and record.get('transcript_id')]
    identities = {(record['versioned_id'], record['structure_sha256']) for record in resolved}
    single = bool(resolved) and len(resolved) == len(records) and len(identities) == 1
    return {'executed_references': resolved,
            'reference_scope': 'single_reference' if single else 'multiple_references'
                               if len(resolved) == len(records) and resolved else 'unresolved',
            'transcript_id': resolved[0]['transcript_id'] if single else None,
            'primary_isoform': resolved[0]['isoform_label'] if single else None}
