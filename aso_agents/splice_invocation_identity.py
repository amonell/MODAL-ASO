"""Content identities for optional reuse inside one splice-evaluation call only."""
from pathlib import Path

from .artifact_io import file_sha256, json_sha256


class InvocationIdentity:
    def __init__(self):
        self._files = {}

    def file_digest(self, path):
        path = Path(path)
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if signature not in self._files:
            self._files[signature] = file_sha256(path)
        return self._files[signature]

    def key(self, *, tiles, goal, transcript, sites, reference, backend, assets):
        # An unresolved live lookup or missing model asset cannot attest reuse.
        from .gapmer_selectivity import resolve_catalog_label
        catalog = goal.get('isoform_transcripts') or {}
        if resolve_catalog_label(transcript, catalog, catalog) is None:
            return None
        try:
            digests = {str(path): self.file_digest(path) for path in [tiles, reference, *assets]}
        except (OSError, TypeError):
            return None
        # The entire goal is passed to the evaluator and its row annotation.
        # Including it is intentionally conservative when job results change.
        return json_sha256({'version': 1, 'files': digests, 'goal': goal,
            'transcript_id': transcript, 'splice_sites': sites, 'backend': backend,
            'pad': 10, 'site_window_bp': 10, 'replacement_motif': 'AAA'})


def evaluation_assets(goal, reference):
    from .splicing_agent import resolve_splice_eval_backend
    from .species_config import get_species_config
    from .splice_eval_assets import required_splicetransformer_files, splicetransformer_root
    backend = resolve_splice_eval_backend(goal.get('species'))
    reference = reference or get_species_config(goal.get('species')).genome_fasta
    if backend.get('backend') == 'splicetransformer':
        root = splicetransformer_root()
        assets = [*required_splicetransformer_files(), *root.glob('*.py'), *root.joinpath('model').rglob('*.py')]
    elif backend.get('backend') == 'openspliceai' and backend.get('model'):
        model = Path(backend['model'])
        assets = list(model.rglob('*')) if model.is_dir() else [model]
        assets = [path for path in assets if path.is_file()]
    else:
        assets = []
    if not assets:
        assets = [Path('/unresolved-splice-evaluation-assets')]
    return reference, backend, assets
