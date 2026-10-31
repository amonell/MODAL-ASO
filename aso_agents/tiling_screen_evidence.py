"""Bind genomic tiling counts to the reference actually configured for mapping."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=128)
def _digest(path, size, mtime_ns, ctime_ns):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def index_identity(prefix):
    """Require all six Bowtie index components; a filename alone is no identity."""
    for extension in ('ebwt', 'ebwtl'):
        paths = [Path(f'{prefix}.{part}.{extension}') for part in ('1', '2', '3', '4', 'rev.1', 'rev.2')]
        if not all(path.is_file() for path in paths):
            continue
        hashes = []
        for path in paths:
            stat = path.stat()
            hashes.append(_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return {'prefix': str(Path(prefix).resolve()), 'format': extension,
                'component_sha256': hashes,
                'sha256': hashlib.sha256(json.dumps(hashes).encode()).hexdigest()}
    return None


def bind_tiling_screen(payload, output_dir, species):
    """Keep per-query censoring while preserving comparable search provenance.

    This binds the generated configuration and full index contents, not a
    clinical safety claim. Completed status still comes from reconciliation of
    the actual mapping records and overflow records for each query.
    """
    config_path = Path(str(Path(output_dir).resolve()) + '.runtime') / 'config.generated.yaml'
    if not config_path.is_file() or not species:
        return
    import yaml
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or Path(str(config.get('OUTPUT_FOLDER') or '')).resolve() != Path(output_dir).resolve():
        return
    budget = config.get('NUM_MISMATCHES')
    if type(budget) is not int or budget not in range(4):
        return
    primary = index_identity(config.get('BOWTIE_INDEX') or '')
    secondary = index_identity(config.get('SECONDARY_INDEX') or '')
    if not primary or not secondary:
        return
    reference = {'species': species, 'assembly': config.get('UCSC_SPECIES'),
        'exact_reference_index': primary, 'substitution_reference_index': secondary,
        'mismatch_budget': budget, 'mapping_mode': 'Bowtie ungapped genomic placements, both strands',
        'count_unit': 'genomic_loci', 'target_exclusion': 'intended genomic placement only',
        'junction_search': 'not performed by genomic tiling',
        'configuration_path': str(config_path),
        'configuration_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    payload['genomic_screen_reference'] = reference
    for row in payload.get('rows') or []:
        if (row.get('Off_target_mismatch_budget') != budget
                or row.get('off_target_search_status') not in {'complete', 'incomplete_or_censored'}):
            continue
        row['off_target_search_provenance_by_species'] = {species: reference}
        row['off_target_screening_status_by_species'] = {species: row['off_target_search_status']}
    columns = payload.setdefault('schema', {}).setdefault('columns', {})
    columns['off_target_search_provenance_by_species'] = 'object'
    columns['off_target_screening_status_by_species'] = 'object'
