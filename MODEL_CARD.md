---
license: bsd-3-clause
model_card_spec: "1.2"
pipeline_tag: other
base_model: torchvision/raft_large
date_published: "2022-03"
date_published_source: "month torchvision 0.12.0 was uploaded to PyPI (2022-03-10); 0.12.0 is the first release whose raft_large builder loads raft_large_C_T_SKHT_V2-ff5fadd5.pth, which torchvision 0.13 kept as Raft_Large_Weights.C_T_SKHT_V2"
---

# RAFT-Large (C_T_SKHT_V2) — Dense Optical Flow with Bounded Fine-Tuning

[![torchvision](https://img.shields.io/badge/torchvision-raft__large-ee4c2c?style=flat&logo=pytorch&logoColor=white)](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-pytorch%2Fvision-181717?style=flat&logo=github&logoColor=white)](https://github.com/pytorch/vision)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2003.12039-b31b1b.svg)](https://arxiv.org/abs/2003.12039)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://github.com/pytorch/vision/blob/main/LICENSE)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> The upstream checkpoint is pinned to the SHA-256 of its bytes, `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`, and the manifest records that digest and the byte size. No execution with the pinned weights has been recorded yet, so this card claims no measured value for this repository.

---

## Interactive Colab Tutorials

- **End-to-end optical flow and adaptation tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/raft-optical-flow-pipeline/blob/main/tutorials/raft_optical_flow_colab.ipynb) [`raft_optical_flow_colab.ipynb`](https://github.com/kurtvalcorza/raft-optical-flow-pipeline/blob/main/tutorials/raft_optical_flow_colab.ipynb)
  *Flow on a rendered pair with exact ground truth, an update-count sweep, static and featureless probes, a bounded fine-tune with RAFT's sequence loss, held-out end-point error before and after next to a zero-flow baseline, new-data inference, and SafeTensors adapter export and reload.*

---

#### Description

`torchvision/raft_large` names torchvision's `raft_large` model with its default weights, `Raft_Large_Weights.C_T_SKHT_V2`. The architecture is RAFT, Recurrent All-Pairs Field Transforms (Teed and Deng, arXiv:2003.12039). torchvision's weight metadata lists 5,257,536 parameters, and a test in this repository counts the same number in the architecture it builds.

RAFT estimates **optical flow**: for each pixel `(x, y)` of frame 1, a displacement `(u, v)` in pixels such that the same scene point appears at `(x + u, y + v)` in frame 2. It has three parts:

- a **feature encoder** that maps each frame to features at 1/8 resolution, and a **context encoder** that maps frame 1 to the initial hidden state and context of the recurrent unit;
- an all-pairs **correlation volume** between every frame-1 feature and every frame-2 feature, pooled into a four-level pyramid;
- a recurrent **update block** that starts from zero flow and refines it a fixed number of times, each time looking up the pyramid around the current estimate. A learned **mask predictor** upsamples each estimate to full resolution.

torchvision's weight documentation says these weights "were trained from scratch", pre-trained on FlyingChairs and FlyingThings3D and then fine-tuned on a combination of Sintel, KITTI, HD1K and FlyingThings3D (clean pass). The training recipe is torchvision's `references/optical_flow`.

This repository adds gradient fine-tuning on a caller's frame pairs with ground-truth flow. `finetune` trains with RAFT's sequence loss. The feature and context encoders are frozen by default, so only the update block and the mask predictor train. Every BatchNorm layer (RAFT-Large has them in the context encoder) is held in evaluation mode, as torchvision's reference does in its Sintel stage with `--freeze-batch-norm`.

What this repository adds to the upstream weights:

- `verify_snapshot` and `stage_missing_files`: manifest checks and staging of the one checkpoint file from its manifest URL, both refusing to run if `MODEL_REVISION` is ever reset to the `"unpinned"` sentinel;
- `RaftPipeline.from_pretrained`: construction with `weights=None` (nothing downloaded by torchvision), then `torch.load(..., weights_only=True)` and `load_state_dict(strict=True)` from the verified file only;
- `estimate`: input checks, torchvision's `OpticalFlow` preprocessing, padding to a multiple of 8 px and cropping back, and a finite `(H, W, 2)` flow array;
- `validate_inputs`, `validate_dataset`, `read_flow_records` and `evaluation_report`: the validation and single-pair evaluation stages, with a zero-flow baseline;
- `flow_metrics`, `evaluate`, `finetune`, `save_artifact`, `apply_artifact` and `load_artifact`: the bounded adaptation workflow and its SafeTensors adapter;
- `samples.py`: rendered frame pairs with exact flow and validity masks, and Middlebury `.flo` reading and writing;
- `tools/pin_snapshot.py`, `tools/build_notebook.py` and `tools/validate_release_assets.py`: pinning, notebook generation and static release checks.

#### Intended Use and Limitations

The uses below are the ones the package was built to support. Everything else is out of scope (Out-of-scope use cases) or prohibited (Use cases).

###### Primary Intended Uses

The task is dense two-frame optical flow. `estimate` takes two `PIL.Image.Image` frames of one size and a number of recurrent updates. It returns a float32 array of shape `(H, W, 2)` holding `(u, v)` in pixels for every pixel of frame 1.

The pretrained weights fit video of natural and rendered scenes: camera motion, moving objects, and the motion statistics of the Sintel, KITTI and HD1K training sets (an animated film and street scenes from a moving car). The adaptation path fits a small set of frame pairs with known flow from a new domain, for example pairs rendered by a simulator.

The intended role is a reference flow estimator and a teaching baseline. RAFT is a standard baseline that later optical-flow methods are compared against. A reader's own application can embed `RaftPipeline` as a first stage of motion analysis, video stabilisation, frame interpolation, motion segmentation or tracking.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, students and instructors. Settings envisioned are research prototypes, teaching, and self-hosted applications that run the code in this repository.

A user is expected to know the following before relying on the output:

- flow is defined from frame 1 to frame 2, and the order of the frames matters;
- flow is an estimate of apparent motion, not of physical motion: a moving shadow or a change of lighting can produce flow, and a textureless moving surface can produce none;
- pixels that are occluded in frame 2, or that leave the frame, have no true correspondence, and the model still returns a value for them;
- end-point error, and every other accuracy figure, can only be measured on pairs with ground-truth flow that the user supplies;
- a fine-tune on a few dozen pairs demonstrates the workflow and does not produce a deployable model.

###### Out-of-scope use cases

1. **Capability boundary:** two frames at a time only; no multi-frame or video-level flow, no stereo disparity, no scene flow, no per-pixel confidence or occlusion output. No object identities or tracks.
2. **Input boundary:** `validate_pair` rejects anything that is not a `PIL.Image.Image` (`TypeError`), two frames of different sizes, sides below `MIN_IMAGE_SIDE = 128` px (torchvision's `min_size` for these weights) or above `MAX_IMAGE_SIDE = 1024` px, and frames over `MAX_PIXELS = 786,432` pixels (1024×768) (`ValueError`). The ceilings are this repository's: the correlation volume grows with the square of the pixel count, and at 1024×768 its finest level alone holds (128·96)² ≈ 151 million float32 values, about 604 MB. `num_flow_updates` must be 1–32.
3. **Input boundary:** motion is limited by what the correlation lookup can reach; displacements of hundreds of pixels are outside what the tutorial measures, and the loss ignores true displacements of `MAX_FLOW = 400` px or more.
4. **Data boundary for adaptation:** `validate_dataset` accepts 1–2,000 records (`MAX_RECORDS`), each with a flow of the frame's shape and an optional boolean valid mask, and `split_pairs` needs at least 2. The bounded tutorial fine-tune (3 epochs by default) is far shorter than the upstream multi-GPU recipe. It is not a way to train a production model.
5. **Decision boundary:** not for decisions that act on flow without a person reviewing them. This covers vehicle control, collision warning, robot navigation, and medical motion analysis. It also requires end-point error measured locally on the deployment's own pairs with ground truth.

#### Factors

###### Groups

The pipeline is not human-centric: it estimates pixel motion and outputs no identity, attribute or label. People appear in video, and their motion is estimated like any other. Neither the upstream authors nor this repository measured whether flow accuracy differs across people, for example by clothing, skin tone or body size under a given lighting; that difference is unknown, not known to be absent.

An adapted model inherits whatever structure the caller's training pairs have. The operator who uses flow on footage of people owns an audit on their own data before relying on it.

###### Instrumentation

The upstream training data is mostly rendered: FlyingChairs and FlyingThings3D are synthetic, and Sintel is rendered from an animated film. KITTI and HD1K are real street footage from car-mounted cameras, with semi-dense ground truth that was measured rather than rendered. Inference frames arrive from whatever produced them: phones, CCTV, dashboard cameras, drones, microscopes or rendering engines.

Frame rate, exposure, motion blur, compression, rolling shutter, lens distortion and sensor noise all change the evidence. The pipeline checks only type, size and that the frames match. It cannot detect a dropped frame, a scene cut, swapped frame order or a frame pair from two different cameras.

The tutorial's sample data is itself an instrument: textured noise moved by whole pixels, with exact flow by construction. Flow a caller supplies carries whatever error its source has, such as simulator approximations or registration error. The fine-tune learns a systematic error of that kind as if it were correct.

###### Environment

**Operating environment.** Python 3.12 with the pins in `pyproject.toml`: `torch==2.14.0`, `torchvision==0.29.0`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`. Computation is float32. The code runs on CPU and uses CUDA automatically when available. torchvision supplies the architecture and the `OpticalFlow` preprocessing. No run with the pinned weights has been recorded yet, so no runtime, memory or throughput figure is given.

**Data environment.** The pretrained model assumes two consecutive frames of one scene with moderate motion. An adapted model assumes inference pairs that resemble its training pairs in camera, scene and motion range. The tutorial's adaptation data is synthetic, so a model adapted on it transfers to rendered textures moving by whole pixels and to nothing else. When these assumptions fail, the model still returns a dense flow field. The pipeline reports no signal that the distribution has shifted.

#### Metrics

###### Performance Measures

`flow_metrics(predicted, truth, valid)` and `evaluate(records)` report, over valid pixels whose true displacement is below `MAX_FLOW`:

- `epe`: the end-point error, the mean Euclidean distance in pixels between the predicted and the true displacement;
- `1px`, `3px` and `5px`: the fraction of pixels whose error is below 1, 3 and 5 px, the thresholds torchvision's reference evaluation reports;
- `angular_error_deg`: the mean angle between the space-time vectors `(u, v, 1)` of prediction and truth, the Middlebury convention;
- `pixel_weighted_epe` (in `evaluate`): EPE with every valid pixel weighted equally, where `epe` weighs every pair equally.

EPE is the benchmark's headline number and is dominated by large errors. The outlier fractions show how much of the image is roughly right. `evaluate` reports every figure next to the **zero-flow baseline**, which predicts no motion; its EPE equals the mean true displacement, so a model is only useful where it beats that row.

`evaluation_report(result, truth, valid)` covers one pair, with the zero-flow baseline and the verdict `sample-sanity`. Without ground truth it returns `not-measurable` and names the data that would be needed.

torchvision's weight metadata reports end-point errors of 1.819 on the Sintel test clean pass and 3.067 on the final pass. Those values are upstream-reported, and this repository does not reproduce them. No value from this repository has been recorded yet.

###### Decision thresholds

The pipeline applies no decision threshold: every pixel gets a flow vector, and nothing is discarded. The one request parameter that changes the output is `num_flow_updates`. Its default of 12 is torchvision's default for `raft_large`; the tutorial compares 1, 4, 12 and 32. More updates cost proportionally more compute.

The 1, 3 and 5 px outlier thresholds are evaluation conventions, not acceptance criteria. No acceptance threshold on EPE is set anywhere in the repository. A deployment that thresholds flow, for example to detect motion, owns choosing and validating that threshold on its own pairs, and re-validating it after any change of camera, frame rate, scene or adapter.

###### Approaches to uncertainty and variability

Every EPE value is one pass over one held-out split: no repeated runs, no cross-validation, no bootstrap, and no confidence interval. The tutorial's held-out split has 10 rendered pairs, so its values are tutorial evidence only.

Sources of run-to-run variability:

- the dataset draw, the split and the new-data draw, controlled by `DATASET_SEED`, `SEED` and `NEW_DATA_SEED`;
- the batch order, controlled by the `seed` argument;
- GPU kernel selection and the `grid_sample` backward pass used by the correlation lookup, which is not forced to be deterministic, so repeated GPU runs can differ slightly.

RAFT outputs no confidence. A caller who needs one per pixel must derive it, for example from forward–backward consistency, and validate it on their own data. A caller who needs an uncertainty estimate for EPE must evaluate over many pairs, with repeated runs or bootstrap resampling.

#### Ethical considerations and biases

No external ethics board, red team, or population-specific review has examined this repository or, to our knowledge, the upstream checkpoint. Nothing below implies that one did.

###### Data

The weights are torchvision's `C_T_SKHT_V2` weights, trained on FlyingChairs, FlyingThings3D, Sintel, KITTI and HD1K as described above. Each of these datasets is distributed under its own terms, and this repository did not review them. KITTI and HD1K are street scenes recorded from a car and can contain people, faces and licence plates; the synthetic sets and Sintel contain no real people.

torchvision's README states that its pre-trained models may have their own licences or terms derived from the dataset used for training, and that the user must determine whether they have permission to use a model for their use case. The BSD-3-Clause licence above is torchvision's code licence; the training-data terms bear on the weights.

This repository distributes code, tests and documentation. It does not distribute the checkpoint, which is staged locally from download.pytorch.org and git-ignored; torchvision's weight metadata lists it as 20.129 MB. It ships no photographs or video: every frame in the tutorial is rendered in code.

An exported adapter contains weights fitted to the caller's training pairs. It does not contain the frames, but it can reflect them. The operator must audit the footage they process, or fine-tune on, for personal, proprietary or restricted content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit or housing. Neither this repository, torchvision's maintainers, nor any regulator has validated or certified it for any of them.

Some sensitive uses are foreseeable although not intended: motion cues for driver assistance or collision warning, motion detection in surveillance video, gait or gesture analysis, and cardiac or respiratory motion estimation in medical imaging. Any of them would be admissible only with human review of every acted-on output. They would also need locally measured end-point error on the deployment's own pairs, a documented re-validation policy, and any regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** while `MODEL_REVISION` is `"unpinned"`, `verify_snapshot`, `stage_missing_files` and `from_pretrained` raise before any download or model import. Once pinned, `stage_missing_files` refuses a manifest whose `modelId` or `revision` differs from the package constants, and a manifest URL that differs from `WEIGHTS_URL`. It fetches only with `allow_download=True`, and `torch.hub.download_url_to_file` checks the file-name SHA-256 prefix on the way in. `verify_snapshot` then checks the file's byte size and full SHA-256 and refuses an entry with no recorded digest. `from_pretrained` builds the architecture with `weights=None`, so torchvision downloads nothing itself, and loads the verified file with `torch.load(..., weights_only=True)` and `load_state_dict(strict=True)`.
- **Tests of those refusals:** tests assert that an unpinned package, a missing checkpoint and a tampered digest are all refused before `torch` or `torchvision` is imported. Others assert that `WEIGHTS_URL` and `MIN_IMAGE_SIDE` equal torchvision's own weight metadata, that the built architecture has the published parameter count, that a frozen fine-tune leaves every encoder weight and BatchNorm statistic unchanged, and that without the BatchNorm hold the statistics do drift.
- **Input integrity:** `validate_inputs` and `estimate` share one checker for type, matching size and size ceilings. `validate_dataset` rejects a record with missing keys, mismatched frames, a flow of the wrong shape or with non-finite values, or a valid mask that is not boolean, has the wrong shape or selects no pixel. `read_flow_records` refuses absolute file names and `..` segments, and `read_flo` refuses a bad header, an implausible size or truncated data. `estimate` raises on a backend flow of the wrong shape or with non-finite values.
- **Adapter integrity:** `save_artifact` writes SafeTensors, not pickle, with the base identity, base-checkpoint digest and frozen prefixes in its header. `apply_artifact` refuses a different format, base identity or model key, a different base digest, tensors the model does not have, and any missing trainable tensor.
- **Reproducibility:** exact `==` pins in `pyproject.toml`, carried into the notebook and checked by the parity tests. Seeds for data, split and batch order. Every result and adapter records `model_id` and `model_revision`.
- **Refusals:** no download without the explicit flag, no unrestricted pickle deserialisation, and no export of an unadapted pipeline.
- **Statistical mitigations:** none is implemented. There is no re-sampling or augmentation; `validate_dataset` reports the displacement range and does not change it.

###### Risks and harms

- **Confident flow where there is none.** On textureless regions, reflections, transparent surfaces and occluded pixels the model returns plausible-looking vectors with no signal that they are guesses. Any system that acts on them inherits the error; the tutorial probes a featureless pair and records the invented motion.
- **Wrong motion at boundaries and for fast objects.** Thin structures, motion blur and large displacements are where flow errors concentrate, and the mean EPE can hide them. Harm falls on whoever relies on the flow of a particular object, such as a pedestrian near a vehicle.
- **Overfitting in adaptation.** A fine-tune on a few dozen pairs can score well on a held-out split drawn from the same source and fail on anything else. The operator who deploys it bears the harm, which is realised whenever training and deployment footage differ.
- **Surveillance.** Motion analysis supports tracking people in video. The people in the processed footage bear the harm of misuse.
- **Automation bias.** A smooth, colour-coded flow field looks precise whether or not it is. Operators who skip review turn a model error into a decision error.
- **Leakage through adaptation data.** A random split of pairs cut from the same video puts near-duplicate frames on both sides. The resulting held-out EPE overstates quality without any signal; `split_pairs` documents that such pairs must be split by video.
- **Resource use.** The correlation volume grows with the square of the pixel count. A large frame or a video stream on a shared host can exhaust memory; the size ceilings bound one call, not a stream.

###### Use cases

The following uses are prohibited even where the model would work:

- tracking, surveilling or profiling people, or supporting biometric identification from motion such as gait;
- unlawful discrimination in employment, housing, credit, insurance, education, healthcare access or law enforcement;
- processing footage the operator has no right to process, or in breach of consent, privacy or data-protection obligations;
- deceptive uses that present estimated motion as measured fact or as evidence;
- autonomous physical control or safety interlocks driven by unreviewed flow;
- any use that violates the upstream BSD-3-Clause licence, the terms attached to the training datasets, or the terms of the deployment running the pipeline.

## Immutable provenance

- Model: `torchvision/raft_large` (torchvision builder `raft_large`, weights `Raft_Large_Weights.C_T_SKHT_V2`, the builder's default).
- Revision: `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`. A URL-hosted file has no commit, so the pinned revision is the SHA-256 of the checkpoint's bytes. `python tools/pin_snapshot.py` pinned it on 2026-09-25: it downloaded the file, checked that its SHA-256 starts with `ff5fadd5` (the prefix in its file name), strict-loaded it into the architecture, and recorded the full digest and the byte size.
- Checkpoint manifest: `weights/raft-large-c-t-skht-v2/dimer-base-manifest.json`, format `dimer_url_snapshot`, one file.
- Checkpoint: `raft_large_C_T_SKHT_V2-ff5fadd5.pth` at `https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth`; 21,106,607 bytes, SHA-256 `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`. torchvision's weight metadata lists the file as 20.129 MB, which agrees.
- Loader: `raft_large(weights=None, progress=False)`, then `load_state_dict(torch.load(<verified file>, map_location="cpu", weights_only=True), strict=True)`.

## Input/output contract

- `RaftPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`: stage (only with `allow_download=True`), verify, load.
- `estimate(image1, image2, *, num_flow_updates=12) -> dict`: keys `flow` (float32 `(H, W, 2)`, `(u, v)` in pixels from frame 1 to frame 2), `num_flow_updates`, `width`, `height`, `adapted`, `model_id`, `model_revision`. Frames are converted to RGB, scaled to `[-1, 1]`, padded by edge replication to a multiple of 8 px, and the flow is cropped back.
- `finetune(records, *, epochs=3, batch_size=2, learning_rate=2e-5, weight_decay=5e-5, num_flow_updates=12, gamma=0.8, seed=20260925, freeze_encoders=True, progress=None) -> dict`: AdamW, gradient clip norm 1.0, float32, BatchNorm held in evaluation mode; batches group frames of one size; returns the configuration, parameter counts and per-epoch losses.
- `evaluate(records, *, num_flow_updates=12) -> dict`: `epe`, `1px`, `3px`, `5px`, `angular_error_deg`, `pixel_weighted_epe`, `n_pairs`, `zero_flow_baseline`, `estimation`.
- `save_artifact(path, *, notes=None) -> dict`; `read_artifact_metadata(path) -> dict`; `apply_artifact(path)`; `load_artifact(path, *, weights_dir=None, device=None)`. The adapter format is `raft-adapter-v1`.
- Records: `{"image1": PIL.Image.Image, "image2": PIL.Image.Image, "flow": float array (H, W, 2), "valid": bool array (H, W), optional}`; `read_flow_records(directory)` reads them from `pairs.json` plus frame, `.flo`/`.npy` and valid-mask files.
- Constants: `MIN_IMAGE_SIDE = 128`, `MAX_IMAGE_SIDE = 1024`, `MAX_PIXELS = 786432`, `MAX_RECORDS = 2000`, `DEFAULT_FLOW_UPDATES = 12`, `MAX_FLOW = 400.0`.

## Verification records

No execution with the pinned weights has been recorded. The offline test suite runs the full `raft_large` architecture with random weights on 128×128 rendered pairs through fine-tuning, evaluation and adapter reload; that exercises the code path and is not a result about this model. `docs/release-verification.md` holds the release gate and the record table.

## References

- Teed and Deng. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. ECCV 2020. https://arxiv.org/abs/2003.12039
- Butler, Wulff, Stanley and Black. A Naturalistic Open Source Movie for Optical Flow Evaluation. ECCV 2012 (MPI-Sintel).
- Baker, Scharstein, Lewis, Roth, Black and Szeliski. A Database and Evaluation Methodology for Optical Flow. IJCV 92, 2011 (Middlebury `.flo` format and angular error).
- torchvision model documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html
- torchvision optical-flow training reference: https://github.com/pytorch/vision/tree/main/references/optical_flow
- Upstream code: https://github.com/pytorch/vision
