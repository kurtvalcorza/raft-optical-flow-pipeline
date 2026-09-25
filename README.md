# RAFT-Large optical flow pipeline

DIMER pipeline for **RAFT-Large with torchvision's `C_T_SKHT_V2` weights** (`torchvision/raft_large`), the recurrent all-pairs model for dense two-frame optical flow. The pipeline loads the checkpoint only from a digest-verified local file, returns a per-pixel `(u, v)` displacement field between two frames, scores it by end-point error against a zero-flow baseline when ground truth is available, and adds a bounded fine-tuning workflow with RAFT's sequence loss that exports a SafeTensors adapter.

> **The upstream checkpoint is pinned** (pinned 2026-09-25) to the SHA-256 of its bytes, `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`. The manifest records the download URL, that digest and the byte size (21,106,607); the digest starts with the `ff5fadd5` prefix in the file name, and the file strict-loaded into the architecture. No execution with the pinned weights is recorded yet (see [Release status](#release-status)).

## Upstream alignment

- Model: `torchvision/raft_large` (builder `raft_large`, weights `Raft_Large_Weights.C_T_SKHT_V2`, the builder's default)
- Revision: `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`, the SHA-256 of the checkpoint file
- Weights host: `https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth` (not the Hugging Face Hub)
- Upstream code license: BSD-3-Clause (torchvision); the training datasets carry their own terms
- Upstream task: dense optical flow, trained on FlyingChairs and FlyingThings3D, then fine-tuned on Sintel, KITTI, HD1K and FlyingThings3D
- Repository adaptation: bounded gradient fine-tuning of the update block and mask predictor, with the encoders and every BatchNorm layer frozen by default

## Quick start

```python
from PIL import Image
from raft_optical_flow_pipeline import RaftPipeline, flow_dataset, flow_to_rgb, split_pairs

pipe = RaftPipeline.from_pretrained(allow_download=True)  # stages + verifies weights/raft-large-c-t-skht-v2
out = pipe.estimate(Image.open("frame_0001.png"), Image.open("frame_0002.png"))
flow = out["flow"]                                          # (H, W, 2) float32, (u, v) in pixels
flow_to_rgb(flow).save("flow.png")                          # hue = direction, saturation = speed

train, held_out = split_pairs(flow_dataset(40), train_fraction=0.75)
adapter = RaftPipeline.from_pretrained()
before = adapter.evaluate(held_out)
print(before["epe"], before["zero_flow_baseline"]["epe"])   # pretrained vs predicting no motion
adapter.finetune(train)                                     # 3 epochs, encoders and BatchNorm frozen
print(adapter.evaluate(held_out)["epe"])                    # adapted
adapter.save_artifact("outputs/raft_adapter.safetensors")
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`, and run `pytest` for the offline test suite. No weights are needed: `tests/test_small_model.py` builds the full architecture with random weights and runs it on 128×128 rendered pairs through fine-tuning, evaluation and adapter reload.

## Pinning the checkpoint

The checkpoint is pinned (see [Upstream alignment](#upstream-alignment)). To re-pin it, from the repository root with network access to download.pytorch.org:

1. Run `python tools/pin_snapshot.py`. It downloads the manifest URL, checks that the file's SHA-256 starts with `ff5fadd5` (the prefix in its name), strict-loads it into `raft_large`, moves it into `weights/raft-large-c-t-skht-v2/`, and writes the digest and byte size into the manifest and the digest into `MODEL_REVISION`.
2. Commit, then run `python tools/build_notebook.py` and commit the regenerated notebook.
3. Update the digest and byte size cited in `README.md`, `MODEL_CARD.md`, `STATUS.md`, `docs/WEIGHTS.md`, `tutorials/README.md` and `docs/release-verification.md`.
4. Run `python tools/validate_release_assets.py` and `pytest`. A new pin invalidates any recorded execution, so the status returns to Candidate until the new checkpoint is run.

## Weights layout

```
weights/raft-large-c-t-skht-v2/
  dimer-base-manifest.json                  # modelId, revision, url, bytes + SHA-256 (1 file)
  raft_large_C_T_SKHT_V2-ff5fadd5.pth       # git-ignored; staged from download.pytorch.org
```

## Input ceilings

`MIN_IMAGE_SIDE = 128` (torchvision's `min_size` for these weights), `MAX_IMAGE_SIDE = 1024` and `MAX_PIXELS = 786432` (1024×768; the correlation volume grows with the square of the pixel count), `num_flow_updates` 1–32 with `DEFAULT_FLOW_UPDATES = 12` (torchvision's default). Frames of one pair must share a size; they are padded to a multiple of 8 px and the flow is cropped back. Adaptation datasets hold 1–2,000 records; true displacements of `MAX_FLOW = 400` px or more are ignored by the loss and the metrics. See `MODEL_CARD.md` for what EPE means and who owns any threshold on flow.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/raft-optical-flow-pipeline/blob/main/tutorials/raft_optical_flow_colab.ipynb)

`tutorials/raft_optical_flow_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.1 and is **standalone** (§4): `tools/build_notebook.py` generates it, and it carries the package modules, the model identity, the manifest and the runtime pins, so it runs without this repository. Its default `Run all` path estimates flow on a rendered pair with exact ground truth, sweeps the number of recurrent updates, probes a static and a featureless pair, validates a 40-pair rendered dataset, measures a baseline, fine-tunes, evaluates the held-out split by end-point error next to a zero-flow baseline, estimates flow on unseen pairs, and exports and reloads the adapter. BYOD frame-pair and dataset branches are off by default. See `tutorials/README.md` and `docs/release-verification.md`.

## Release status

**Candidate.** The checkpoint is pinned (SHA-256 `ff5fadd56d26…`), but no execution with the pinned weights is recorded. Static checks, unit tests and the small-model test do not constitute notebook execution evidence; `docs/release-verification.md` defines the release gate.

## Documentation

- `MODEL_CARD.md`: MODEL_CARD_SPEC 1.2 card, provenance, input/output contract.
- `docs/WEIGHTS.md`: weight provenance, pinning and hosting notes.
- `STATUS.md`: release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). torchvision, which defines the architecture and publishes the weights, is BSD-3-Clause; its README notes that pre-trained weights may carry terms derived from their training data. See `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
