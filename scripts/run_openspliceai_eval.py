#!/usr/bin/env python3
"""
Batch wrapper to score reference and perturbed sequences with OpenSpliceAI and
return splice-site delta metrics for ASO simulations.

Input JSON (via --input or stdin):
{
  "model_path": ".../OSAI_mouse_10000nt_model1.pt",
  "flanking_size": 10000,
  "requests": [
    {
      "id": "tile_0",
      "reference_sequence": "ACGT...",
      "deletion_sequence": "ACNN...",
      "replacement_sequence": "ACAAA...",
      "sites": [
        {
          "key": "site_0",          # opaque identifier
          "acceptor_index": 123,    # 0-based index in oriented sequence
          "donor_index": 145,
          "window": 50
        },
        ...
      ],
      "primary": {                 # fallback indices when sites list is empty
        "acceptor_index": 123,
        "donor_index": 145,
        "window": 50
      }
    }
  ]
}

Output JSON (to stdout or --output):
{
  "results": [
    {
      "id": "tile_0",
      "length": 512,
      "metrics": {
        "deletion": {"pair": -0.12, "acceptor": -0.08, "donor": -0.18},
        "replacement": {"pair": -0.05, "acceptor": -0.03, "donor": -0.09}
      },
      "sites": [
        {
          "key": "site_0",
          "deletion": {"pair": -0.11, "acceptor": -0.07, "donor": -0.17},
          "replacement": {"pair": -0.04, "acceptor": -0.02, "donor": -0.08}
        },
        ...
      ]
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
import sys
import importlib.util
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from openspliceai.predict import predict as predict_mod  # type: ignore
from openspliceai.predict import utils as predict_utils  # type: ignore


@lru_cache(maxsize=1)
def _cryptic_calculator():
    # A separately installed model interpreter must not import the harness's
    # compiled dependencies or package initializer. This helper needs only
    # NumPy from this interpreter and is shipped beside the native package.
    script = Path(__file__).resolve()
    candidates = [script.parents[1] / "aso_agents" / "cryptic_splice_evidence.py",
                  script.parents[2] / "cryptic_splice_evidence.py"]
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise FileNotFoundError("Packaged cryptic-splice evidence helper is missing")
    spec = importlib.util.spec_from_file_location("modal_cryptic_splice_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.paired_evidence


@dataclass
class EvalRequest:
    req_id: str
    reference_sequence: str
    deletion_sequence: str
    replacement_sequence: str
    sites: List[Dict[str, int]]
    primary: Optional[Dict[str, int]]
    cryptic_context: Optional[dict] = None


def _parse_requests(payload: Dict[str, object]) -> Tuple[str, int, List[EvalRequest]]:
    model_path = str(payload.get("model_path") or "").strip()
    if not model_path:
        raise ValueError("model_path missing in payload")
    flanking_size = int(payload.get("flanking_size") or 10000)
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("requests must be a non-empty list")

    requests: List[EvalRequest] = []
    for idx, item in enumerate(raw_requests):
        if not isinstance(item, dict):
            raise ValueError(f"request[{idx}] must be an object")
        req_id = str(item.get("id") or f"request_{idx}")
        ref_seq = str(item.get("reference_sequence") or "").upper()
        del_seq = str(item.get("deletion_sequence") or "").upper()
        rep_seq = str(item.get("replacement_sequence") or "").upper()
        if not ref_seq or not del_seq or not rep_seq:
            raise ValueError(f"request[{idx}] sequences must be non-empty")

        sites_raw = item.get("sites") or []
        sites: List[Dict[str, int]] = []
        if not isinstance(sites_raw, Sequence):
            raise ValueError(f"request[{idx}].sites must be a list when provided")
        for site_idx, site in enumerate(sites_raw):
            if not isinstance(site, dict):
                raise ValueError(f"request[{idx}].sites[{site_idx}] must be an object")
            try:
                acceptor_index = int(site["acceptor_index"])
                donor_index = int(site["donor_index"])
            except Exception as exc:
                raise ValueError(f"request[{idx}].sites[{site_idx}] missing indices: {exc}") from exc
            window = int(site.get("window", 50))
            sites.append(
                {
                    "key": str(site.get("key") or f"site_{site_idx}"),
                    "acceptor_index": acceptor_index,
                    "donor_index": donor_index,
                    "window": window,
                    "acceptor_window": int(site.get("acceptor_window", window)),
                    "donor_window": int(site.get("donor_window", window)),
                }
            )

        primary_raw = item.get("primary")
        primary: Optional[Dict[str, int]] = None
        if isinstance(primary_raw, dict):
            try:
                primary = {
                    "acceptor_index": int(primary_raw["acceptor_index"]),
                    "donor_index": int(primary_raw["donor_index"]),
                    "window": int(primary_raw.get("window", 50)),
                    "acceptor_window": int(primary_raw.get("acceptor_window", primary_raw.get("window", 50))),
                    "donor_window": int(primary_raw.get("donor_window", primary_raw.get("window", 50))),
                }
            except Exception:
                primary = None

        requests.append(
            EvalRequest(
                req_id=req_id,
                reference_sequence=ref_seq,
                deletion_sequence=del_seq,
                replacement_sequence=rep_seq,
                sites=sites,
                primary=primary,
                cryptic_context=item.get("cryptic_context") if isinstance(item.get("cryptic_context"),dict) else None,
            )
        )

    return model_path, flanking_size, requests


class OpenSpliceAIModel:
    """Thin wrapper that caches OpenSpliceAI model weights for repeated inference."""

    def __init__(self, model_path: str, flanking_size: int) -> None:
        self.consts = predict_utils.initialize_constants(flanking_size)
        self.device = predict_mod.setup_device()
        self.models, _ = predict_mod.load_pytorch_models(
            model_path, self.device, self.consts["SL"], flanking_size
        )

    def predict(self, sequence: str) -> np.ndarray:
        X = predict_mod.create_datapoints(sequence, self.consts["SL"], self.consts["CL_max"])
        tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        DNAs = tensor.to(self.device)
        with torch.no_grad():
            stacked = torch.stack([mdl(DNAs).detach().cpu() for mdl in self.models], dim=0)
            mean_pred = torch.mean(stacked, dim=0)
        flat = mean_pred.permute(0, 2, 1).contiguous().view(-1, mean_pred.shape[1])
        length = len(sequence)
        return flat[:length, :].numpy()


@lru_cache(maxsize=1)
def _measurement_calculator():
    # The model worker can run in an interpreter without harness dependencies.
    import importlib.util
    script = Path(__file__).resolve()
    candidates = [script.parents[1] / "aso_agents" / "splice_measurement_windows.py",
                  script.parents[2] / "splice_measurement_windows.py"]
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise FileNotFoundError("Packaged splice measurement helper is missing")
    spec = importlib.util.spec_from_file_location("modal_splice_measurement_windows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.delta_metrics


def _compute_metrics(pred_ref, pred_alt, acc_idx, don_idx, window, donor_window=None):
    return _measurement_calculator()(pred_ref, pred_alt, acc_idx, don_idx, window, donor_window)


def evaluate_requests(model_path: str, flanking_size: int, requests: List[EvalRequest]) -> Dict[str, object]:
    model = OpenSpliceAIModel(model_path, flanking_size)
    results: List[Dict[str, object]] = []

    for req in requests:
        pred_ref = model.predict(req.reference_sequence)
        pred_del = model.predict(req.deletion_sequence)
        pred_rep = model.predict(req.replacement_sequence)

        site_results: List[Dict[str, object]] = []
        for site in req.sites:
            acc_idx = site["acceptor_index"]
            don_idx = site["donor_index"]
            acc_window = site.get("acceptor_window", site["window"])
            don_window = site.get("donor_window", site["window"])
            del_metrics = _compute_metrics(pred_ref, pred_del, acc_idx, don_idx, acc_window, don_window)
            rep_metrics = _compute_metrics(pred_ref, pred_rep, acc_idx, don_idx, acc_window, don_window)
            site_results.append(
                {
                    "key": site["key"],
                    "deletion": del_metrics,
                    "replacement": rep_metrics,
                }
            )

        primary_metrics: Dict[str, Dict[str, Optional[float]]] = {
            "deletion": {"pair": None, "acceptor": None, "donor": None},
            "replacement": {"pair": None, "acceptor": None, "donor": None},
        }
        if req.primary:
            acc_idx = req.primary["acceptor_index"]
            don_idx = req.primary["donor_index"]
            acc_window = req.primary.get("acceptor_window", req.primary["window"])
            don_window = req.primary.get("donor_window", req.primary["window"])
            primary_metrics["deletion"] = _compute_metrics(pred_ref, pred_del, acc_idx, don_idx, acc_window, don_window)
            primary_metrics["replacement"] = _compute_metrics(pred_ref, pred_rep, acc_idx, don_idx, acc_window, don_window)

        results.append(
            {
                "id": req.req_id,
                "length": int(pred_ref.shape[0]),
                "metrics": primary_metrics,
                "sites": site_results,
                **({"cryptic_splice_evidence":_cryptic_calculator()(
                    pred_ref,pred_del,pred_ref,pred_rep,model="OpenSpliceAI",
                    first_perturbation="length_preserving_N_mask",**req.cryptic_context)} if req.cryptic_context else {}),
            }
        )

    return {"results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenSpliceAI on reference and ASO-perturbed sequences.")
    parser.add_argument("--input", type=str, help="Path to JSON payload (defaults to stdin).")
    parser.add_argument("--output", type=str, help="Optional output path (defaults to stdout).")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.load(sys.stdin)

    model_path, flanking_size, requests = _parse_requests(payload)
    output_payload = evaluate_requests(model_path, flanking_size, requests)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, ensure_ascii=False)
    else:
        json.dump(output_payload, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
