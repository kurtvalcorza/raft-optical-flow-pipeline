"""Deterministic in-code frame pairs with exact optical flow, and Middlebury ``.flo`` input/output.

Nothing here is downloaded and nothing needs torch — numpy and Pillow only. Each pair is rendered from a
larger textured canvas:

* the **background** is a window into the canvas; frame 2's window is shifted by a whole number of pixels,
  so every background pixel moves by the same known displacement;
* each **object** is a textured disc or square pasted at one position in frame 1 and at a displaced
  position in frame 2, so its pixels move by that object's own known displacement.

The flow is defined from frame 1 to frame 2: the pixel at ``(x, y)`` in frame 1 appears at
``(x + u, y + v)`` in frame 2. A pixel is **valid** when its correspondence is visible in frame 2: object
pixels always are (objects stay inside the frame and do not overlap), background pixels are not when an
object covers their destination or when the destination leaves the frame. Scores are computed on valid
pixels only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

PAIR_SIZE = (256, 256)  # (width, height), a multiple of 8 as RAFT requires
MAX_OBJECTS = 4
FLO_MAGIC = 202021.25  # Middlebury .flo header tag


def _texture(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Band-limited colour noise: enough structure for matching at every scale RAFT looks at."""
    base = rng.integers(0, 256, (height // 8 + 2, width // 8 + 2, 3), dtype=np.uint8)
    coarse = np.asarray(Image.fromarray(base).resize((width, height), Image.BICUBIC), dtype=np.float32)
    fine = rng.integers(0, 64, (height, width, 3)).astype(np.float32)
    return np.clip(0.8 * coarse + fine, 0, 255).astype(np.uint8)


def _shape_mask(kind: str, radius: int) -> np.ndarray:
    side = 2 * radius + 1
    yy, xx = np.mgrid[:side, :side]
    if kind == "disc":
        return (xx - radius) ** 2 + (yy - radius) ** 2 <= radius**2
    return np.ones((side, side), dtype=bool)


def moving_shapes_pair(
    seed: int = 0,
    *,
    size: tuple[int, int] = PAIR_SIZE,
    max_background_shift: int = 6,
    max_object_shift: int = 12,
    n_objects: int | None = None,
) -> dict[str, Any]:
    """One frame pair with its exact flow ``(H, W, 2)`` float32 (u, v) and boolean valid mask ``(H, W)``."""
    width, height = size
    if width % 8 or height % 8 or min(width, height) < 128:
        raise ValueError(f"size must be multiples of 8 and at least 128 px, got {size}")
    rng = np.random.default_rng(seed)
    margin = max_background_shift
    canvas = _texture(rng, height + 2 * margin, width + 2 * margin)
    bx, by = (int(v) for v in rng.integers(-max_background_shift, max_background_shift + 1, 2))
    frame1 = canvas[margin : margin + height, margin : margin + width].copy()
    frame2 = canvas[margin + by : margin + by + height, margin + bx : margin + bx + width].copy()
    # A window shifted by (bx, by) shows the content moved by (-bx, -by).
    flow = np.zeros((height, width, 2), dtype=np.float32)
    flow[..., 0], flow[..., 1] = -bx, -by
    ys, xs = np.mgrid[:height, :width]
    valid = (xs - bx >= 0) & (xs - bx < width) & (ys - by >= 0) & (ys - by < height)

    count = int(rng.integers(1, MAX_OBJECTS + 1)) if n_objects is None else n_objects
    placed: list[tuple[int, int, int]] = []
    covered2 = np.zeros((height, width), dtype=bool)
    objects = []
    attempts = 0
    while len(objects) < count and attempts < 200:
        attempts += 1
        radius = int(rng.integers(14, 34))
        dx, dy = (int(v) for v in rng.integers(-max_object_shift, max_object_shift + 1, 2))
        lo = radius + max_object_shift + 1
        cx, cy = int(rng.integers(lo, width - lo)), int(rng.integers(lo, height - lo))
        reach = radius + max_object_shift + 2
        if any(abs(cx - px) < reach + pr and abs(cy - py) < reach + pr for px, py, pr in placed):
            continue
        placed.append((cx, cy, radius))
        kind = "disc" if rng.random() < 0.5 else "square"
        mask = _shape_mask(kind, radius)
        patch = _texture(rng, mask.shape[0], mask.shape[1])
        y0, x0 = cy - radius, cx - radius
        region1 = (slice(y0, y0 + mask.shape[0]), slice(x0, x0 + mask.shape[1]))
        region2 = (slice(y0 + dy, y0 + dy + mask.shape[0]), slice(x0 + dx, x0 + dx + mask.shape[1]))
        frame1[region1][mask] = patch[mask]
        frame2[region2][mask] = patch[mask]
        flow[region1][mask] = (dx, dy)
        valid[region1] |= mask
        covered2[region2] |= mask
        objects.append({"kind": kind, "center": [cx, cy], "radius": radius, "shift": [dx, dy]})

    # A background pixel whose destination is covered by an object in frame 2 is occluded.
    object_pixels = np.zeros((height, width), dtype=bool)
    for obj in objects:
        cx, cy, r = obj["center"][0], obj["center"][1], obj["radius"]
        m = _shape_mask(obj["kind"], r)
        object_pixels[cy - r : cy + r + 1, cx - r : cx + r + 1] |= m
    dest_x = np.clip(xs - bx, 0, width - 1)
    dest_y = np.clip(ys - by, 0, height - 1)
    occluded = ~object_pixels & covered2[dest_y, dest_x]
    valid &= ~occluded
    return {
        "id": f"pair-{seed}",
        "image1": Image.fromarray(frame1),
        "image2": Image.fromarray(frame2),
        "flow": flow,
        "valid": valid,
        "background_shift": [-bx, -by],
        "objects": objects,
    }


def flow_dataset(
    n_pairs: int = 40, *, seed: int = 0, size: tuple[int, int] = PAIR_SIZE
) -> list[dict[str, Any]]:
    """A deterministic list of ``moving_shapes_pair`` records, the adaptation dataset of the tutorial."""
    if not 1 <= n_pairs <= 1000:
        raise ValueError(f"n_pairs must be in 1..1000, got {n_pairs}")
    return [moving_shapes_pair(seed * 100_000 + i, size=size) for i in range(n_pairs)]


def split_pairs(
    records: list[dict[str, Any]], *, train_fraction: float = 0.75, seed: int = 0
) -> tuple[list, list]:
    """Seeded random split at the pair level; pairs cut from the same video must be split by video instead."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")
    if len(records) < 2:
        raise ValueError(f"at least 2 records are required to split, got {len(records)}")
    order = np.random.default_rng(seed).permutation(len(records))
    n_train = min(len(records) - 1, max(1, int(len(records) * train_fraction)))
    chosen = {int(i) for i in order[:n_train]}
    return [r for i, r in enumerate(records) if i in chosen], [
        r for i, r in enumerate(records) if i not in chosen
    ]


def static_pair(size: tuple[int, int] = PAIR_SIZE, seed: int = 7) -> dict[str, Any]:
    """The same textured frame twice: the true flow is zero everywhere."""
    width, height = size
    frame = Image.fromarray(_texture(np.random.default_rng(seed), height, width))
    return {
        "id": "static",
        "image1": frame,
        "image2": frame.copy(),
        "flow": np.zeros((height, width, 2), np.float32),
        "valid": np.ones((height, width), bool),
    }


def blank_pair(size: tuple[int, int] = PAIR_SIZE) -> dict[str, Any]:
    """Two featureless white frames: any motion is undetermined, and a flow estimate is a guess."""
    width, height = size
    frame = Image.new("RGB", size, (255, 255, 255))
    return {
        "id": "blank",
        "image1": frame,
        "image2": frame.copy(),
        "flow": np.zeros((height, width, 2), np.float32),
        "valid": np.ones((height, width), bool),
    }


def read_flo(path: Any) -> np.ndarray:
    """Read a Middlebury ``.flo`` file into an ``(H, W, 2)`` float32 array; refuse a bad header or size."""
    with open(path, "rb") as fh:
        magic = np.fromfile(fh, np.float32, count=1)
        if magic.size != 1 or magic[0] != FLO_MAGIC:
            raise ValueError(f"{path}: not a Middlebury .flo file (bad magic)")
        width, height = (int(v) for v in np.fromfile(fh, np.int32, count=2))
        if not (0 < width <= 4096 and 0 < height <= 4096):
            raise ValueError(f"{path}: implausible .flo size {width}x{height}")
        data = np.fromfile(fh, np.float32, count=2 * width * height)
    if data.size != 2 * width * height:
        raise ValueError(f"{path}: truncated .flo data")
    return data.reshape(height, width, 2)


def write_flo(path: Any, flow: np.ndarray) -> None:
    """Write an ``(H, W, 2)`` flow array as a Middlebury ``.flo`` file."""
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"flow must have shape (H, W, 2), got {flow.shape}")
    with open(path, "wb") as fh:
        np.array([FLO_MAGIC], np.float32).tofile(fh)
        np.array([flow.shape[1], flow.shape[0]], np.int32).tofile(fh)
        flow.tofile(fh)
