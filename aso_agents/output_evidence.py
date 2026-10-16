"""Keep complete output-source evidence on disk without duplicating it in prompts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def store_reference_projection(projection: list[dict], directory: Path) -> dict:
    """Publish the exact recorded observations under their content digest."""
    data = (json.dumps(projection, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()
    digest = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + '.json')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Recorded output evidence changed: ' + str(path))
    else:
        with path.open('xb') as handle:
            handle.write(data)
    return {'path': str(path.resolve()), 'sha256': digest,
            'format': 'JSON array of recorded input observations',
            'observation_count': len(projection),
            'complete_recorded_projection': True,
            'scope': 'All retained reference groups and alignment records, including records outside the current requested reference or search criteria. Filter by matching identities and scope before comparing counts; this catalog is not an exhaustive search guarantee.'}
