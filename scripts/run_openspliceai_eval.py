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
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from openspliceai.predict import predict as predict_mod  # type: ignore
from openspliceai.predict import utils as predict_utils  # type: ignore


@dataclass
class EvalRequest:
    req_id: str
    reference_sequence: str
    deletion_sequence: str
    replacement_sequence: str
    sites: List[Dict[str, int]]
    primary: Optional[Dict[str, int]]


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
            window = int(site.get("window") or 50)
            sites.append(
                {
                    "key": str(site.get("key") or f"site_{site_idx}"),
                    "acceptor_index": acceptor_index,
                    "donor_index": donor_index,
                    "window": window,
                }
            )

        primary_raw = item.get("primary")
        primary: Optional[Dict[str, int]] = None
        if isinstance(primary_raw, dict):
            try:
                primary = {
                    "acceptor_index": int(primary_raw["acceptor_index"]),
                    "donor_index": int(primary_raw["donor_index"]),
                    "window": int(primary_raw.get("window") or 50),
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


def _window_max(arr: np.ndarray, index: int, window: int, column: int) -> float:
    if index < 0 or index >= arr.shape[0]:
        return 0.0
    lo = max(0, index - window)
    hi = min(arr.shape[0], index + window + 1)
    if hi <= lo:
        return 0.0
    return float(np.max(arr[lo:hi, column]))


def _compute_metrics(
    pred_ref: np.ndarray,
    pred_alt: np.ndarray,
    acc_idx: int,
    don_idx: int,
    window: int,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {"pair": None, "acceptor": None, "donor": None}
    if pred_ref.shape[0] == 0 or pred_alt.shape[0] == 0:
        return metrics

    acc_ref = _window_max(pred_ref, acc_idx, window, column=1)
    acc_alt = _window_max(pred_alt, acc_idx, window, column=1)
    don_ref = _window_max(pred_ref, don_idx, window, column=2)
    don_alt = _window_max(pred_alt, don_idx, window, column=2)

    metrics["acceptor"] = acc_alt - acc_ref
    metrics["donor"] = don_alt - don_ref
    I_ref = math.sqrt(max(acc_ref, 0.0) * max(don_ref, 0.0))
    I_alt = math.sqrt(max(acc_alt, 0.0) * max(don_alt, 0.0))
    metrics["pair"] = I_alt - I_ref
    return metrics


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
            window = site["window"]
            del_metrics = _compute_metrics(pred_ref, pred_del, acc_idx, don_idx, window)
            rep_metrics = _compute_metrics(pred_ref, pred_rep, acc_idx, don_idx, window)
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
            window = req.primary["window"]
            primary_metrics["deletion"] = _compute_metrics(pred_ref, pred_del, acc_idx, don_idx, window)
            primary_metrics["replacement"] = _compute_metrics(pred_ref, pred_rep, acc_idx, don_idx, window)

        results.append(
            {
                "id": req.req_id,
                "length": int(pred_ref.shape[0]),
                "metrics": primary_metrics,
                "sites": site_results,
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
