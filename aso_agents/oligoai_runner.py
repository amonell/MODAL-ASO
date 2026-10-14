"""Isolated released-model inference. Run through aso_agents.oligoai."""
import argparse
from contextlib import nullcontext
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .oligoai import (SOURCE_REVISION, MODEL_SHA256, MODEL_FILE, METHODS,
                     file_sha, validate_model_input)
from .response_artifacts import digest, write_json


def predict(inputs, root, *, batch_size=16, device="auto"):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not inputs:
        raise ValueError("No model inputs")
    for item in inputs:
        validate_model_input(item)
    root = Path(root).resolve()
    checkpoint, source = root / MODEL_FILE, root / "source"
    if file_sha(checkpoint) != MODEL_SHA256:
        raise ValueError("Unrecognized checkpoint; do not deserialize it")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("OligoAI source is not the pinned unmodified release")
    sys.path.insert(0, str(source))
    import torch
    from train_aso import ASOInhibitionPredictionWrapper
    from rinalmo.data.alphabet import Alphabet
    import rinalmo.model.attention as attention
    import flash_attn
    original_unpad = attention.unpad_input
    def compatible_unpad(*args, **kwargs):
        values = original_unpad(*args, **kwargs)
        # FlashAttention 2.7 adds a fifth auxiliary used-length vector. RiNALMo
        # consumes only the same four original outputs; tensors are unchanged.
        if len(values) not in (4, 5):
            raise RuntimeError("Unknown FlashAttention unpad_input interface")
        return values[:4]
    attention.unpad_input = compatible_unpad
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError("Unsupported inference device")
    if device != "cuda":
        raise RuntimeError("The released OligoAI giga checkpoint requires CUDA FlashAttention")
    # Same host-global lock as SpliceTransformer. Wait, never assign a scientific
    # failure merely because another process owns the device.
    from .paths import shared_gpu_lock
    lock_path = shared_gpu_lock()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    inherited = os.environ.get("ASO_OLIGOAI_GPU_LOCK_FD")
    if inherited is not None:
        fd = int(inherited)
        stat, expected = os.fstat(fd), lock_path.stat()
        if (stat.st_dev, stat.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("Inherited GPU descriptor is not the shared ownership lock")
        lock = os.fdopen(os.dup(fd), "a+")
    else:
        lock = lock_path.open("a+") if device == "cuda" else nullcontext()
    started = time.monotonic()
    with lock as handle:
        if device == "cuda":
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        # The checksum above pins the only trusted pickle admitted by this runner.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = ASOInhibitionPredictionWrapper(**saved["hyper_parameters"])
        model.load_state_dict(saved["state_dict"], strict=True)
        del saved
        model.eval().to(device)
        alphabet = Alphabet()
        predictions = []
        vocab = {"DNA": 1, "MOE": 2, "cET": 3}
        bonds = {"PO": 1, "PS": 2}
        with torch.inference_mode():
            for offset in range(0, len(inputs), batch_size):
                batch = inputs[offset:offset + batch_size]
                length = max(len(i["antisense_5to3"]) for i in batch) + 2
                context_length = max(len(i["rna_context"]) for i in batch) + 2
                seqs = [alphabet.encode(i["antisense_5to3"].replace("T", "U"), pad_to_len=length) for i in batch]
                contexts = [alphabet.encode(i["rna_context"], pad_to_len=context_length) for i in batch]
                # Official ASODataset: CLS gets modification pad; outgoing bonds
                # follow each nucleotide, with a final nucleotide pad (n-1 bonds).
                sugars, backbone = [], []
                for item in batch:
                    chem = [0] + [vocab[s] for s in item["sugar_mods"]]
                    link = [0] + [bonds[s] for s in item["backbone_mods"]]
                    sugars.append(chem + [0] * (length - len(chem)))
                    backbone.append(link + [0] * (length - len(link)))
                tensors = [torch.tensor(v, dtype=torch.long, device=device) for v in (seqs, sugars, backbone, contexts)]
                tensors[3][tensors[3] == alphabet.unk_idx] = alphabet.mask_idx
                dose = torch.tensor([i["dosage_nm"] for i in batch], dtype=torch.float32, device=device)
                method = torch.tensor([METHODS[i["transfection_method"]] for i in batch], dtype=torch.long, device=device)
                autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
                with autocast:
                    scaled = model(*tensors, dose, method)
                    values = model.scaler.inverse_transform(scaled).float().cpu().tolist()
                predictions.extend({"input_id": item["input_id"], "predicted_inhibition_percent": float(value)} for item, value in zip(batch, values))
        metadata = {"model_sha256": MODEL_SHA256, "source_revision": SOURCE_REVISION,
                    "input_digest": digest(inputs), "runner_sha256": file_sha(__file__),
                    "torch_version": torch.__version__, "device": device, "batch_size": batch_size,
                    "duration_seconds": time.monotonic() - started,
                    "tokenization": "released ASODataset vocabularies; n-1 outgoing bonds then terminal pad",
                    "precision": "float16_autocast" if device == "cuda" else "float32"}
        metadata["flash_attention_version"] = flash_attn.__version__
        metadata["compatibility"] = "unpad_input auxiliary fifth return ignored; original four tensors preserved"
        if device == "cuda":
            metadata["device_name"] = torch.cuda.get_device_name(0)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return {**metadata, "predictions": predictions}


def main():
    import json
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if payload.get("schema_version") != "modal-aso-oligoai-input/1.0":
        raise ValueError("Unsupported OligoAI input schema")
    result = predict(payload["inputs"], args.root, batch_size=args.batch_size, device=args.device)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
