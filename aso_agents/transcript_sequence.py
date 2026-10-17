"""Reference cDNA retrieval for operations that do not require a local genome."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4
import requests

from aso_agents.species_config import ENSEMBL_REST, USER_AGENT


def reference_cdna(transcript_id, expected_length=None):
    identifier = str(transcript_id or "").strip()
    if not identifier:
        raise ValueError("transcript_identifier_missing")
    url = f"{ENSEMBL_REST}/sequence/id/{quote(identifier, safe='')}"
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "modal-aso" / "transcript-sequences"
    key = hashlib.sha256((url + ':cdna').encode()).hexdigest()
    cache_path = cache_root / f"{key}.json"
    evidence = None
    if cache_path.exists():
        evidence = json.loads(cache_path.read_text())
    if evidence is None:
        response = requests.get(url, params={"type": "cdna", "content-type": "application/json"},
                                headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        response.raise_for_status()
        evidence = {"source_url": response.url, "response": response.json(), "transcript_id": identifier}
    record = evidence["response"]
    if str(record.get("id") or "").split(".")[0] != identifier.split(".")[0]:
        raise ValueError("transcript_sequence_identity_mismatch")
    sequence = str(record.get("seq") or "").upper().replace("U", "T")
    if not sequence or set(sequence) - set("ACGTN"):
        raise ValueError("invalid_reference_cdna")
    if expected_length is not None and len(sequence) != expected_length:
        raise ValueError("reference_cdna_length_disagrees_with_exon_structure")
    evidence["sequence_sha256"] = hashlib.sha256(sequence.encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(evidence, indent=2) + '\n')
    temporary.replace(cache_path)
    return sequence, {"source_url": evidence["source_url"], "record_path": str(cache_path),
                      "sequence_sha256": evidence["sequence_sha256"], "transcript_id": record["id"],
                      "length": len(sequence), "method": "ensembl_reference_cdna"}
