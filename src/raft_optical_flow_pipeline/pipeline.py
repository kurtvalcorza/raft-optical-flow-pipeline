"""Dense optical flow and bounded fine-tuning with torchvision's RAFT-Large (``C_T_SKHT_V2`` weights).

The model is torchvision's own ``raft_large`` architecture, built with ``weights=None`` so that nothing is
downloaded at construction, then loaded from one digest-verified local checkpoint file (``weights/<key>/``)
with ``torch.load(..., weights_only=True)``, which refuses arbitrary pickled objects.

Until ``tools/pin_snapshot.py`` has recorded the checkpoint's SHA-256 and byte size, the package refuses to
stage, verify or load weights: an unpinned checkpoint is never trusted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .samples import read_flo

MODEL_ID = "torchvision/raft_large"
# The pinned identity of a URL-hosted checkpoint is the SHA-256 of its bytes. torchvision names the file
# after the first 8 hex digits of that digest, so the URL is content-addressed; the full digest is pinned.
MODEL_REVISION = "unpinned"
MODEL_LICENSE = "bsd-3-clause"
MODEL_KEY = "raft-large-c-t-skht-v2"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
ARTIFACT_FORMAT = "raft-adapter-v1"
UNPINNED = "unpinned"
PIN_COMMAND = "python tools/pin_snapshot.py"
WEIGHTS_FILE = "raft_large_C_T_SKHT_V2-ff5fadd5.pth"
WEIGHTS_URL = f"https://download.pytorch.org/models/{WEIGHTS_FILE}"
WEIGHTS_SHA256_PREFIX = "ff5fadd5"
TORCHVISION_WEIGHTS = "Raft_Large_Weights.C_T_SKHT_V2"

# Operational ceilings. RAFT's all-pairs correlation volume holds (H/8 * W/8)^2 values, so memory grows with
# the square of the pixel count: 1024 x 768 already needs about 0.6 GB for it alone.
MIN_IMAGE_SIDE = 128  # torchvision's weight metadata: min_size (128, 128)
MAX_IMAGE_SIDE = 1024
MAX_PIXELS = 1024 * 768
MAX_RECORDS = 2000
DEFAULT_FLOW_UPDATES = 12  # torchvision's default num_flow_updates
MAX_FLOW = 400.0  # the reference sequence loss ignores ground-truth displacements at or above this (pixels)

# Bounded fine-tuning defaults, following torchvision's optical-flow reference (references/optical_flow):
# AdamW, sequence loss with gamma 0.8, gradient-norm clipping at 1.0, BatchNorm frozen as in its Sintel stage.
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 5e-5
DEFAULT_GAMMA = 0.8
DEFAULT_TRAIN_UPDATES = 12
DEFAULT_SEED = 20260925
ENCODER_PREFIXES = ("feature_encoder.", "context_encoder.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pinned() -> bool:
    """True once MODEL_REVISION is the checkpoint's 64-hex SHA-256."""
    revision = MODEL_REVISION
    return len(revision) == 64 and all(c in "0123456789abcdef" for c in revision)


def _require_pinned(action: str) -> None:
    if not is_pinned():
        raise RuntimeError(
            f"refusing to {action}: {MODEL_ID} has no pinned checkpoint digest yet (MODEL_REVISION = "
            f"{MODEL_REVISION!r}); run `{PIN_COMMAND}` to record the SHA-256 and byte size"
        )


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local checkpoint against its DIMER manifest; raise naming the first mismatch."""
    _require_pinned("verify the checkpoint")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        if not entry.get("sha256") or entry.get("bytes") is None:
            raise ValueError(f"{entry['path']}: manifest records no sha256/bytes; run `{PIN_COMMAND}`")
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"checkpoint file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _url_download(entry: Mapping[str, Any], root: Path) -> None:
    """Fetch one manifest entry from its URL; torch.hub checks the filename's SHA-256 prefix on the way in."""
    from torch.hub import download_url_to_file

    download_url_to_file(
        entry["url"], str(root / entry["path"]), hash_prefix=WEIGHTS_SHA256_PREFIX, progress=False
    )


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[Mapping[str, Any], Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the checkpoint). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    _require_pinned("stage the checkpoint")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"checkpoint at {root} is missing {[e['path'] for e in missing]}; "
            "pass allow_download=True to fetch it"
        )
    fetch = downloader or _url_download
    for entry in missing:
        if entry.get("url") != WEIGHTS_URL:
            raise ValueError(f"manifest URL {entry.get('url')!r} != package URL {WEIGHTS_URL!r}")
        fetch(entry, root)
    return [entry["path"] for entry in missing]


# --------------------------------------------------------------------------- metrics


def flow_metrics(predicted: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    """End-point error (EPE), outlier rates and angular error of a flow field against ground truth.

    ``EPE`` is the mean Euclidean distance in pixels between predicted and true displacement over valid
    pixels. ``1px``/``3px``/``5px`` are the fractions of valid pixels whose error is below that many pixels
    (higher is better), as torchvision's reference evaluation reports them. ``angular_error_deg`` is the mean
    angle between the space-time vectors (u, v, 1), the Middlebury convention.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape or truth.ndim != 3 or truth.shape[2] != 2:
        raise ValueError(f"flows must both have shape (H, W, 2), got {predicted.shape} and {truth.shape}")
    mask = np.ones(truth.shape[:2], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != truth.shape[:2]:
        raise ValueError(f"valid mask shape {mask.shape} != flow shape {truth.shape[:2]}")
    mask = mask & (np.linalg.norm(truth, axis=2) < MAX_FLOW)
    if not mask.any():
        raise ValueError("no valid pixel to score")
    error = np.linalg.norm(predicted - truth, axis=2)[mask]
    p, t = predicted[mask], truth[mask]
    # Angle between (u, v, 1) vectors as atan2(|a x b|, a . b): exact near 0, where arccos is not.
    a = np.column_stack([p, np.ones(len(p))])
    b = np.column_stack([t, np.ones(len(t))])
    angle = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(axis=1))
    return {
        "epe": float(error.mean()),
        "1px": float((error < 1).mean()),
        "3px": float((error < 3).mean()),
        "5px": float((error < 5).mean()),
        "angular_error_deg": float(np.degrees(angle).mean()),
        "valid_pixels": int(mask.sum()),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean of per-pair metrics (each pair weighs the same) plus the pixel-weighted EPE."""
    if not rows:
        raise ValueError("no rows to aggregate")
    keys = ("epe", "1px", "3px", "5px", "angular_error_deg")
    out: dict[str, Any] = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    pixels = sum(row["valid_pixels"] for row in rows)
    out["pixel_weighted_epe"] = float(sum(row["epe"] * row["valid_pixels"] for row in rows) / pixels)
    out["n_pairs"] = len(rows)
    return out


# --------------------------------------------------------------------------- input validation


def validate_pair(image1: Any, image2: Any) -> tuple[Image.Image, Image.Image]:
    """Two PIL images of one size inside the ceilings; returns them converted to RGB."""
    for name, image in (("image1", image1), ("image2", image2)):
        if not isinstance(image, Image.Image):
            raise TypeError(f"{name} must be a PIL.Image.Image, got {type(image).__name__}")
    if image1.size != image2.size:
        raise ValueError(f"frames differ in size: {image1.size} vs {image2.size}")
    width, height = image1.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"frame side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"frame side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    if width * height > MAX_PIXELS:
        raise ValueError(f"frame has {width * height} pixels > MAX_PIXELS {MAX_PIXELS}")
    return image1.convert("RGB"), image2.convert("RGB")


def _check_updates(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32:
        raise ValueError(f"num_flow_updates must be an int in 1..32, got {value!r}")
    return value


INPUT_SCHEMA: dict[str, Any] = {
    "input": "two PIL.Image.Image frames of the same size (any mode, converted to RGB); frame 1 then frame 2",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "max_pixels": MAX_PIXELS,
    "num_flow_updates": [1, 32],
    "output": "flow (H, W, 2) float32 in pixels: frame-1 pixel (x, y) moves to (x + u, y + v) in frame 2",
    "preprocessing": (
        "RGB to float in [0, 1], normalised to [-1, 1] (torchvision's OpticalFlow preset); frames are padded "
        "by edge replication to a multiple of 8 px and the flow is cropped back"
    ),
}


def validate_inputs(
    image1: Any, image2: Any, *, num_flow_updates: int = DEFAULT_FLOW_UPDATES
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict)."""
    first, second = validate_pair(image1, image2)
    updates = _check_updates(num_flow_updates)
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": name, "mode": img.mode, "size": list(img.size)}
            for name, img in (("frame1", image1), ("frame2", image2))
        ],
        "num_flow_updates": updates,
        "padding": [(-first.width) % 8, (-first.height) % 8],
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def validate_dataset(records: Sequence[Mapping[str, Any]], *, epochs: int = DEFAULT_EPOCHS) -> dict[str, Any]:
    """Validation stage for flow records ``{"image1", "image2", "flow", "valid"?}``: raise on the first broken
    record, else return the dataset manifest with displacement statistics."""
    if not records:
        raise ValueError("dataset must hold at least one record")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset holds {len(records)} records > MAX_RECORDS {MAX_RECORDS}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
        raise ValueError(f"epochs must be an int in 1..100, got {epochs!r}")
    magnitudes, valid_fraction, sizes = [], [], set()
    findings = []
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping) or not {"image1", "image2", "flow"} <= set(record):
            raise ValueError(f"record {idx} must be a mapping with 'image1', 'image2' and 'flow'")
        try:
            first, _second = validate_pair(record["image1"], record["image2"])
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"record {idx}: {exc}") from exc
        flow = np.asarray(record["flow"])
        if flow.shape != (first.height, first.width, 2):
            raise ValueError(f"record {idx}: flow shape {flow.shape} != ({first.height}, {first.width}, 2)")
        if not np.isfinite(flow).all():
            raise ValueError(f"record {idx}: flow contains non-finite values")
        valid = np.asarray(record.get("valid", np.ones(flow.shape[:2], bool)))
        if valid.shape != flow.shape[:2] or (valid.dtype != bool and not np.isin(valid, (0, 1)).all()):
            raise ValueError(f"record {idx}: valid mask must be boolean with shape {flow.shape[:2]}")
        valid = valid.astype(bool)
        if not valid.any():
            raise ValueError(f"record {idx}: valid mask selects no pixel")
        magnitude = np.linalg.norm(flow, axis=2)[valid]
        if (magnitude >= MAX_FLOW).any():
            findings.append(
                f"record {idx}: {int((magnitude >= MAX_FLOW).sum())} valid pixels move >= {MAX_FLOW} px "
                "and are ignored by the loss"
            )
        magnitudes.append(float(magnitude.mean()))
        valid_fraction.append(float(valid.mean()))
        sizes.add(first.size)
    if len(sizes) > 1:
        findings.append(f"{len(sizes)} different frame sizes; batches are grouped by size")
    return {
        "n_records": len(records),
        "frame_sizes": sorted([list(s) for s in sizes]),
        "mean_displacement_px": float(np.mean(magnitudes)),
        "max_mean_displacement_px": float(np.max(magnitudes)),
        "mean_valid_fraction": float(np.mean(valid_fraction)),
        "epochs": epochs,
        "findings": findings,
        "verdict": "accepted",
    }


def read_flow_records(directory: str | Path) -> list[dict[str, Any]]:
    """Read BYOD flow records from ``<directory>/pairs.json`` plus the files it names.

    ``pairs.json`` is a list of objects such as
    ``{"frame1": "a.png", "frame2": "b.png", "flow": "a.flo", "valid": "a_valid.png"}``; ``flow`` is a
    Middlebury ``.flo`` file or a ``.npy`` array of shape (H, W, 2), and ``valid`` is optional (a
    single-channel PNG whose non-zero pixels are valid). File names must stay inside ``directory``.
    """
    root = Path(directory).resolve()
    index = root / "pairs.json"
    if not index.is_file():
        raise FileNotFoundError(f"{index} not found; expected pairs.json next to the frames")
    entries = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("pairs.json must hold a non-empty list of records")
    if len(entries) > MAX_RECORDS:
        raise ValueError(f"pairs.json lists {len(entries)} records > MAX_RECORDS {MAX_RECORDS}")

    def resolve(idx: int, name: Any) -> Path:
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"pairs.json entry {idx}: file {name!r} must be relative to {root}")
        resolved = (root / relative).resolve()
        if root not in resolved.parents:
            raise ValueError(f"pairs.json entry {idx}: file {name!r} resolves outside {root}")
        return resolved

    records = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or not {"frame1", "frame2", "flow"} <= set(entry):
            raise ValueError(f"pairs.json entry {idx} must have 'frame1', 'frame2' and 'flow'")
        paths = {key: resolve(idx, entry[key]) for key in ("frame1", "frame2", "flow")}
        frames = []
        for key in ("frame1", "frame2"):
            with Image.open(paths[key]) as handle:
                frames.append(handle.convert("RGB"))
        flow = (
            np.load(paths["flow"], allow_pickle=False)
            if paths["flow"].suffix == ".npy"
            else read_flo(paths["flow"])
        )
        record = {
            "id": str(entry["frame1"]),
            "image1": frames[0],
            "image2": frames[1],
            "flow": flow.astype(np.float32),
        }
        if entry.get("valid"):
            with Image.open(resolve(idx, entry["valid"])) as handle:
                record["valid"] = np.asarray(handle.convert("L")) > 0
        records.append(record)
    return records


def evaluation_report(
    result: Mapping[str, Any],
    truth: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Single-pair evaluation stage: EPE, outlier rates and angular error against ground truth, next to
    the zero-flow baseline (predicting no motion), or ``not-measurable`` without ground truth."""
    flow = np.asarray(result["flow"])
    base = {
        "task": "dense optical flow between two frames",
        "sample_kind": sample_kind,
        "flow_shape": list(flow.shape),
        "num_flow_updates": result.get("num_flow_updates", DEFAULT_FLOW_UPDATES),
        "mean_predicted_displacement_px": float(np.linalg.norm(flow, axis=2).mean()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if truth is None:
        return {
            **base,
            "metrics": {},
            "baselines": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth flow was supplied for the evaluated pair",
            "needs": (
                "frame pairs from your own footage with ground-truth flow (rendered or measured), "
                "scored by EPE and outlier rates against the zero-flow baseline"
            ),
        }
    metrics = flow_metrics(flow, truth, valid)
    baseline = flow_metrics(np.zeros_like(np.asarray(truth, dtype=np.float32)), truth, valid)
    return {
        **base,
        "metrics": metrics,
        "baselines": [{"id": "zero-flow", **baseline}],
        "verdict": "sample-sanity",
        "reason": "one frame pair with exact ground truth; geometry sanity evidence, not a benchmark",
        "needs": "a labelled set of frame pairs from the deployment domain for any EPE claim",
    }


# --------------------------------------------------------------------------- model


def build_model() -> Any:
    """torchvision's raft_large architecture with random weights and nothing downloaded."""
    from torchvision.models.optical_flow import raft_large

    return raft_large(weights=None, progress=False)


def _to_tensor(images: Sequence[Image.Image]) -> Any:
    """torchvision's OpticalFlow preset: uint8 RGB to float in [0, 1], then normalised to [-1, 1]."""
    import torch

    arrays = np.stack([np.asarray(image, dtype=np.float32) / 255.0 for image in images])
    return torch.from_numpy(arrays).permute(0, 3, 1, 2) * 2.0 - 1.0


def _pad8(batch: Any) -> tuple[Any, tuple[int, int]]:
    import torch.nn.functional as F

    height, width = batch.shape[-2:]
    pad_h, pad_w = (-height) % 8, (-width) % 8
    if pad_h or pad_w:
        batch = F.pad(batch, (0, pad_w, 0, pad_h), mode="replicate")
    return batch, (height, width)


def _hold_batchnorm(module: Any) -> None:
    """Keep every BatchNorm layer under ``module`` in eval mode (RAFT's context encoder carries BatchNorm)."""
    import torch

    for sub in module.modules():
        if isinstance(sub, torch.nn.modules.batchnorm._BatchNorm):
            sub.eval()


def sequence_loss(predictions: Sequence[Any], truth: Any, valid: Any, gamma: float = DEFAULT_GAMMA) -> Any:
    """RAFT's loss: L1 flow error over every refinement, weighted gamma**(N - i), valid and < MAX_FLOW px only
    (torchvision references/optical_flow utils.sequence_loss)."""
    import torch

    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    norm = torch.sum(truth**2, dim=1).sqrt()
    mask = (valid & (norm < MAX_FLOW))[:, None, :, :]
    stacked = torch.stack(list(predictions))
    per_iteration = ((stacked - truth).abs() * mask).mean(dim=(1, 2, 3, 4))
    weights = gamma ** torch.arange(
        len(predictions) - 1, -1, -1, device=truth.device, dtype=per_iteration.dtype
    )
    return (per_iteration * weights).sum()


@dataclass
class RaftPipeline:
    """Dense optical flow, and bounded fine-tuning on a caller's flow pairs, over RAFT-Large."""

    model: Any
    device: str
    source: str = "snapshot"
    base_state_digest: str | None = None
    adapted: bool = False
    frozen_prefixes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> RaftPipeline:
        _require_pinned("load the model")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no checkpoint manifest at {root}; stage {MODEL_ID} under weights/{MODEL_KEY} "
                "(allow_download=True fetches the manifest-listed file)"
            )
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)

        import torch

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_model()
        state = torch.load(root / WEIGHTS_FILE, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return cls(
            model=model.to(resolved_device).eval(),
            device=resolved_device,
            source=str(root),
            base_state_digest=MODEL_REVISION,
        )

    def _forward(
        self, first: Sequence[Image.Image], second: Sequence[Image.Image], updates: int
    ) -> list[Any]:
        batch1, (height, width) = _pad8(_to_tensor(first).to(self.device))
        batch2, _ = _pad8(_to_tensor(second).to(self.device))
        predictions = self.model(batch1, batch2, num_flow_updates=updates)
        return [flow[..., :height, :width] for flow in predictions]

    def estimate(
        self, image1: Image.Image, image2: Image.Image, *, num_flow_updates: int = DEFAULT_FLOW_UPDATES
    ) -> dict[str, Any]:
        """Estimate the flow from ``image1`` to ``image2``; returns ``flow`` as an (H, W, 2) float32 array."""
        import torch

        first, second = validate_pair(image1, image2)
        updates = _check_updates(num_flow_updates)
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            flow = self._forward([first], [second], updates)[-1][0]
        if was_training:
            self.model.train()
        array = flow.permute(1, 2, 0).float().cpu().numpy()
        if array.shape != (first.height, first.width, 2) or not np.isfinite(array).all():
            raise RuntimeError(f"backend returned a malformed flow of shape {array.shape}")
        return {
            "flow": array,
            "num_flow_updates": updates,
            "width": first.width,
            "height": first.height,
            "adapted": self.adapted,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def evaluate(
        self, records: Sequence[Mapping[str, Any]], *, num_flow_updates: int = DEFAULT_FLOW_UPDATES
    ) -> dict[str, Any]:
        """Mean EPE, outlier rates and angular error over labelled pairs, next to the zero-flow baseline."""
        rows, zero_rows = [], []
        for record in records:
            flow = self.estimate(record["image1"], record["image2"], num_flow_updates=num_flow_updates)[
                "flow"
            ]
            truth = np.asarray(record["flow"], dtype=np.float32)
            valid = record.get("valid")
            rows.append(flow_metrics(flow, truth, valid))
            zero_rows.append(flow_metrics(np.zeros_like(truth), truth, valid))
        return {
            **aggregate_metrics(rows),
            "zero_flow_baseline": aggregate_metrics(zero_rows),
            "num_flow_updates": num_flow_updates,
            "adapted": self.adapted,
            "estimation": (
                f"one pass over {len(records)} pairs; each pair weighs the same; "
                "no resampling, no dispersion estimate"
            ),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        num_flow_updates: int = DEFAULT_TRAIN_UPDATES,
        gamma: float = DEFAULT_GAMMA,
        seed: int = DEFAULT_SEED,
        freeze_encoders: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning with RAFT's sequence loss, mutating this pipeline.

        Every BatchNorm layer is held in eval mode, as torchvision's reference does in its Sintel stage
        (``--freeze-batch-norm``). With ``freeze_encoders`` the feature and context encoders also keep their
        weights and only the recurrent update block and the upsampling mask predictor train. Gradients are
        clipped to norm 1.0. Frames in a batch must share a size.
        """
        import torch

        validate_dataset(records, epochs=epochs)
        updates = _check_updates(num_flow_updates)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, int | float)
            or not 0.0 < float(learning_rate) <= 1.0
        ):
            raise ValueError(f"learning_rate must be a number in (0, 1], got {learning_rate!r}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not (freeze_encoders and name.startswith(ENCODER_PREFIXES))
        self.frozen_prefixes = ENCODER_PREFIXES if freeze_encoders else ()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=float(weight_decay))

        by_size: dict[tuple[int, int], list[int]] = {}
        for i, record in enumerate(records):
            by_size.setdefault(record["image1"].size, []).append(i)
        epoch_losses: list[float] = []
        for epoch in range(epochs):
            self.model.train()
            _hold_batchnorm(self.model)
            batches = []
            for indices in by_size.values():
                order = [indices[int(j)] for j in rng.permutation(len(indices))]
                batches.extend(order[k : k + batch_size] for k in range(0, len(order), batch_size))
            running, n_batches = 0.0, 0
            for b in rng.permutation(len(batches)):
                batch = [records[i] for i in batches[int(b)]]
                first = [validate_pair(r["image1"], r["image2"])[0] for r in batch]
                second = [validate_pair(r["image1"], r["image2"])[1] for r in batch]
                truth = (
                    torch.from_numpy(np.stack([np.asarray(r["flow"], np.float32) for r in batch]))
                    .permute(0, 3, 1, 2)
                    .to(self.device)
                )
                valid = torch.from_numpy(
                    np.stack(
                        [np.asarray(r.get("valid", np.ones(r["flow"].shape[:2], bool)), bool) for r in batch]
                    )
                ).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = sequence_loss(self._forward(first, second, updates), truth, valid, gamma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                running += float(loss.detach().cpu())
                n_batches += 1
            epoch_losses.append(running / max(1, n_batches))
            if progress is not None:
                progress({"epoch": epoch + 1, "epochs": epochs, "loss": epoch_losses[-1]})
        self.model.eval()
        self.adapted = True
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "optimizer": "AdamW",
            "loss": f"RAFT sequence loss (L1, gamma {gamma}, displacements >= {MAX_FLOW:.0f} px ignored)",
            "gradient_clip_norm": 1.0,
            "batchnorm": "frozen (eval mode)",
            "num_flow_updates": updates,
            "seed": seed,
            "precision": "float32",
            "freeze_encoders": freeze_encoders,
            "frozen_prefixes": list(self.frozen_prefixes),
            "trainable_parameters": sum(p.numel() for p in trainable),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "device": self.device,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def save_artifact(self, path: str | Path, *, notes: str | None = None) -> dict[str, Any]:
        """Write the adapted tensors as one SafeTensors file with the provenance in its metadata.

        Tensors under ``frozen_prefixes`` are left out: they equal the verified base checkpoint, which
        ``load_artifact`` loads first. The artifact is therefore an adapter bound to the base checkpoint.
        """
        from safetensors.torch import save_file

        if not self.adapted:
            raise RuntimeError("nothing to export: the pipeline has not been fine-tuned")
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in self.model.state_dict().items()
            if not any(name.startswith(prefix) for prefix in self.frozen_prefixes)
        }
        metadata = {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_key": MODEL_KEY,
            "frozen_prefixes": json.dumps(list(self.frozen_prefixes)),
            "base_state_digest": self.base_state_digest or "",
            "notes": notes or "",
        }
        save_file(tensors, str(artifact_path), metadata=metadata)
        return {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
            "format": ARTIFACT_FORMAT,
            "tensors": len(tensors),
            "frozen_prefixes": list(self.frozen_prefixes),
            "base_state_digest": self.base_state_digest,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @staticmethod
    def read_artifact_metadata(path: str | Path) -> dict[str, Any]:
        """Read and check the artifact's provenance header without loading any tensor."""
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            metadata = dict(handle.metadata() or {})
        if metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {metadata.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if (metadata.get("model_id"), metadata.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError(
                f"artifact was built on {metadata.get('model_id')}@{metadata.get('model_revision')}, "
                f"package pins {MODEL_ID}@{MODEL_REVISION}"
            )
        if metadata.get("model_key") != MODEL_KEY:
            raise ValueError(
                f"artifact was built on {metadata.get('model_key')!r}, package pins {MODEL_KEY!r}"
            )
        return {**metadata, "frozen_prefixes": json.loads(metadata["frozen_prefixes"])}

    def apply_artifact(self, path: str | Path) -> None:
        """Load adapter tensors onto this base pipeline; refuse any tensor set that does not fit."""
        from safetensors.torch import load_file

        metadata = self.read_artifact_metadata(path)
        expected_base = metadata.get("base_state_digest") or None
        if expected_base and self.base_state_digest and expected_base != self.base_state_digest:
            raise ValueError("artifact was exported against a different base checkpoint digest")
        tensors = load_file(str(path), device="cpu")
        prefixes = tuple(metadata["frozen_prefixes"])
        result = self.model.load_state_dict(tensors, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                f"artifact carries tensors the model does not have: {result.unexpected_keys[:5]}"
            )
        stray = [key for key in result.missing_keys if not any(key.startswith(p) for p in prefixes)]
        if stray:
            raise ValueError(f"artifact is missing trainable tensors: {stray[:5]}")
        self.model.eval()
        self.adapted = True
        self.frozen_prefixes = prefixes
        self.source = f"artifact:{Path(path).name}"

    @classmethod
    def load_artifact(
        cls, path: str | Path, *, weights_dir: str | Path | None = None, device: str | None = None
    ) -> RaftPipeline:
        """Rebuild an adapted pipeline: verified base checkpoint first, then the adapter tensors."""
        cls.read_artifact_metadata(path)
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir)
        pipe.apply_artifact(path)
        return pipe


def flow_to_rgb(flow: np.ndarray) -> Image.Image:
    """Colour-code a flow field (hue = direction, saturation = speed) with torchvision's flow_to_image."""
    import torch
    from torchvision.utils import flow_to_image

    tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(flow, dtype=np.float32).transpose(2, 0, 1)))
    return Image.fromarray(flow_to_image(tensor).permute(1, 2, 0).numpy())
