"""Publish original materialized site coordinates separately from tiling windows."""
import hashlib
import json
from pathlib import Path


def export_region_sources(goal, run_dir):
    materialized = goal.get('materialized_region_constraints') or {}
    scopes = [('global', None, materialized.get('global') or [])]
    scopes += [('program', program, constraints) for program, constraints
               in (materialized.get('by_program') or {}).items()]
    folder = Path(run_dir) / 'artifacts/design/region_sources'
    records = []
    for scope, program, constraints in scopes:
        for constraint in constraints:
            rows = constraint.get('materialized_rows')
            if not isinstance(rows, list):
                continue
            lines = []
            for row in rows:
                if (not isinstance(row, (list, tuple)) or len(row) < 3
                        or type(row[1]) is not int or type(row[2]) is not int
                        or not 0 <= row[1] < row[2]
                        or any('\t' in str(v) or '\n' in str(v) or '\r' in str(v) for v in row)):
                    raise ValueError('Cannot export an invalid materialized source interval as BED.')
                lines.append('\t'.join(map(str, row)) + '\n')
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'constraint_{len(records)+1:03d}.bed'
            path.write_text(''.join(lines))
            records.append({'scope': scope, 'program_id': program,
                            'constraint_id': constraint.get('constraint_id'),
                            'label': constraint.get('label'),
                            'coordinate_space': constraint.get('coordinate_space') or 'genomic',
                            'coordinate_convention': 'bed0_half_open',
                            'genome_build': constraint.get('genome_build'),
                            'row_count': len(rows), 'rows': rows,
                            'path': str(path),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {'role': 'unmodified_materialized_source_intervals',
              'description': 'Source BED intervals retain their coordinates, order and labels. Tiling search windows are separate artifacts and may include overlap flanks.',
              'records': records}
    if records:
        (folder / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
