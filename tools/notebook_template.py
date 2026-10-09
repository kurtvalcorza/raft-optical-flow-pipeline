"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.2 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
Every value a reader can change is a `# @param` form field, and each file-reading BYOD branch has a
location field that bypasses the upload dialog when set (EXE1, EXE2).

The checkpoint is torchvision's, hosted on download.pytorch.org rather than the Hugging Face Hub, so the
template declares `weights_host`.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "raft_optical_flow_pipeline",
    "repo_name": "raft-optical-flow-pipeline",
    "stem": "raft_optical_flow",
    "notebook_name": "raft_optical_flow_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # SWP-R (2026-10-05 fleet sweep): nothing is pip-installed into the notebook kernel. The fleet's uv isolated-environment
    # mechanism (build_notebook.py/2.2): managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the
    # pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt` (uv 0.12.15).
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "RaftPipeline",
    "guided": {
        "opening": [
            '**Who this notebook is for.** The intended audience is a learner who knows basic Python, PIL and NumPy, has used Colab or Jupyter, and wants to see '
            'how a dense optical-flow model is scored against exact ground truth, where its estimates go wrong, and what a bounded fine-tune of its update block '
            'changes. No prior experience with RAFT or optical flow is assumed; terms are explained where they first matter and again in the **Glossary** at the '
            'end. A GPU runtime (T4) is the documented runtime; CPU works, slowly.\n\n**Input → Model → Output.**\n\n| | Flow estimation | Bounded fine-tune |\n|---|---|---|\n| '
            'Input | two frames of one size (PIL images) | rendered frame pairs with exact flow and validity masks (40 by default: 30 training, 10 held out) |\n| Model | '
            "torchvision's RAFT-Large (`C_T_SKHT_V2`), a digest-pinned checkpoint, 12 recurrent updates | the same model; only the recurrent update block and "
            'upsampling mask predictor train (3.1 M of 5.3 M parameters), encoders and BatchNorm frozen |\n| Output | an `(H, W, 2)` flow field in pixels; EPE, '
            '1/3/5 px rates and angular error beside a zero-flow baseline | held-out EPE before and after, flow on unseen pairs, and a SafeTensors adapter that '
            'reloads to the same flow |\n\n**How to use this notebook.** Choose a GPU runtime (**Runtime → Change runtime type → T4 GPU**), then **Runtime → Run '
            "all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are "
            '**infrastructure** — the isolated environment, the carried package and the pinned checkpoint — and their cells are collapsed; you may run them without '
            'studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the '
            'recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your '
            'reasoning** with a worked answer from the recorded run (the Kaggle Tesla T4 run of 25 September 2026 recorded in `docs/release-verification.md`; GPU '
            'kernels are not deterministic, so your last digits may differ). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. '
            'Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the pretrained model on a pair with exact flow *(core concept: EPE and '
            'the zero-flow baseline)* → 5 recurrent updates and degenerate probes *(evaluation practice)* → 6 the labelled dataset and validation → 7 split and '
            'pre-adaptation baseline *(evaluation practice)* → 8 bounded fine-tuning *(core concept)* → 9 held-out evaluation → 10 unseen pairs → 11 adapter export '
            'and reload *(engineering)* → 12 outputs → 13 optional BYOD → interpretation, troubleshooting, glossary and your conclusion.'
        ],
    },
    "weights_key": "raft-large-c-t-skht-v2",
    "weights_host": {
        "name": "download.pytorch.org",
        "size": "one file that torchvision's weight metadata lists as 20.129 MB",
    },
    "modules": [
        "samples.py",
        "pipeline.py",
    ],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "torchvision", "numpy", "PIL"],
    "title": "RAFT-Large (C_T_SKHT_V2) — DIMER dense optical flow and bounded fine-tuning (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/raft-optical-flow-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/raft-optical-flow-pipeline/blob/main/tutorials/raft_optical_flow_colab.ipynb",
        ),
        (
            "torchvision",
            "https://img.shields.io/badge/torchvision-raft__large-ee4c2c?style=flat&logo=pytorch&logoColor=white",
            "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-pytorch%2Fvision-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/pytorch/vision",
        ),
        (
            "arXiv",
            "https://img.shields.io/badge/arXiv-2003.12039-b31b1b.svg",
            "https://arxiv.org/abs/2003.12039",
        ),
        (
            "License",
            "https://img.shields.io/badge/License-Apache--2.0-green.svg",
            "https://github.com/kurtvalcorza/raft-optical-flow-pipeline/blob/main/LICENSE",
        ),
    ],
    "capability": "dense optical flow between two frames with torchvision's RAFT-Large, and a bounded fine-tune on frame pairs with known flow that is scored on a held-out split by end-point error against a zero-flow baseline and exported as a reloadable SafeTensors adapter",
    "intro": (
        "Optical flow is the apparent motion of every pixel between two frames: for each pixel `(x, y)` of frame 1, a displacement `(u, v)` "
        "in pixels such that the same scene point appears at `(x + u, y + v)` in frame 2. RAFT (Recurrent All-Pairs Field Transforms) "
        "computes it in three parts. A **feature encoder** turns each frame into features at 1/8 resolution; RAFT correlates every feature of "
        "frame 1 with every feature of frame 2 into a 4-D correlation volume; and a **recurrent update block** starts from zero flow and "
        "refines it a fixed number of times, looking up the volume around the current estimate at each step. A learned upsampler returns the "
        "flow at full resolution.\n\n"
        "This notebook uses torchvision's `raft_large` builder with its default weights, `Raft_Large_Weights.C_T_SKHT_V2`. torchvision's weight "
        "documentation says these weights were trained from scratch on FlyingChairs and FlyingThings3D, then fine-tuned on a mix of Sintel, "
        "KITTI, HD1K and FlyingThings3D (clean pass), and lists Sintel test end-point errors of 1.819 (clean pass) and 3.067 (final pass).\n\n"
        "**The default path really adapts the model:** it builds a labelled dataset of textured frame pairs with exact flow, measures the "
        "pretrained end-point error on a held-out split next to a zero-flow baseline, runs a bounded fine-tune with RAFT's own sequence loss, "
        "re-scores the held-out split, runs the adapted model on unseen pairs, exports the changed tensors as a SafeTensors adapter, and "
        "reloads that adapter onto a fresh copy of the verified base model to check that it reproduces the same flow. Every number you see is "
        "measured in this notebook runtime."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the pinned checkpoint; estimate flow "
        "on a frame pair with exact ground truth and score it by end-point error (EPE), outlier rates and angular error against the zero-flow "
        "baseline; see how the number of recurrent updates changes the estimate; probe the model with a static and a featureless pair; "
        "build and validate a labelled flow dataset; split it and measure a pre-adaptation baseline; run a bounded fine-tune with RAFT's "
        "sequence loss; score the adapted model on the held-out split; run inference on unseen pairs; and export, reload and verify the adapter."
    ),
    "exclusions": (
        "real-video optical flow claims (the adaptation dataset is rendered in code with integer displacements, so the model learns these "
        "renderings and nothing about camera footage); Sintel or KITTI benchmark results (the metrics helper follows the benchmark "
        "definitions but is run on synthetic data only); full-schedule RAFT training (torchvision's reference recipe trains for hundreds of "
        "thousands of steps on several GPUs; the tutorial runs a few epochs on 30 pairs by default); stereo disparity; scene flow; video interpolation."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). A CUDA GPU such as a Colab or Kaggle T4 is the documented runtime for the fine-tuning stages and is used automatically when present; the notebook also runs on CPU, much more slowly. Measured with the defaults: on the recorded Kaggle Tesla T4 run of 25 September 2026 the stages after setup took about 35 s (the fine-tune cell 8.9 s); a local CPU run of the same stages took about 139 s (the fine-tune about 96 s; review probe, 4 October 2026). Building the isolated environment the first time (the pinned `torch==2.14.0` wheel is its largest download; reused on a re-run) is the slowest setup step and adds a few minutes.",
        "- **Knowledge:** basic Python, PIL and NumPy; images as `(H, W)` pixel grids; a flow field as an `(H, W, 2)` array of `(u, v)` displacements in pixels; and the Euclidean distance between two vectors.",
        "- **Data:** the default path generates everything in code with `samples.py` and downloads no dataset: one 256×256 demonstration pair and a labelled dataset (40 pairs by default, `N_PAIRS`), each with exact flow and a validity mask. BYOD is optional and off by default. Expected BYOD input: two frames of one size, or a directory holding `pairs.json` — a list of `{'frame1': 'a.png', 'frame2': 'b.png', 'flow': 'a.flo', 'valid': 'a_valid.png'}` objects, where `flow` is a Middlebury `.flo` file or a `.npy` array of shape `(H, W, 2)` and `valid` is optional — and the files it names. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the pinned checkpoint, "
        "estimates flow on a rendered pair with exact ground truth, validates the labelled dataset (40 pairs by default), splits it into training and held-out parts, "
        "measures the pre-adaptation baseline, **runs the bounded fine-tune**, re-evaluates on the held-out split, estimates flow on unseen "
        "pairs, exports the adapter, reloads it onto a fresh base model to verify the flow, and writes machine-readable outputs with provenance. "
        "Nothing is skipped behind a default-off flag, and no clone or DIMER worker is required (NOTEBOOK_SPEC 2.2 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two optional BYOD branches are included, and both are off by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own two frames through the same validation, estimation and evaluation-report stages as the sample pair. "
        "`USE_BYOD_DATASET` takes your own frame pairs with ground-truth flow through the full adaptation workflow — validate, split, baseline, "
        "fine-tune, evaluate, export and reload — under NOTEBOOK_SPEC 2.2 DAT14. Set `BYOD_IMAGE1_PATH` and `BYOD_IMAGE2_PATH`, or "
        "`BYOD_DATASET_DIR`, to read from a location without an upload dialog (EXE2)."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. Demonstration pair
        {
            "md": (
                "## 4. What the pretrained model does on a pair with exact flow\n\n"
                "Before adapting anything, inspect the model you start from. The carried `samples` module renders a deterministic frame pair "
                "from a textured canvas. The **background** moves by one whole-pixel displacement, and each textured disc or square **object** "
                "moves by its own. Because the pair is rendered, the true flow of every pixel is known exactly. A pixel is **valid** when its "
                "destination is visible in frame 2; background pixels that an object covers in frame 2, or that leave the frame, are not.\n\n"
                "The metrics are the ones optical-flow benchmarks report, computed over valid pixels only:\n"
                "- **EPE** (end-point error): the mean distance in pixels between the predicted and the true displacement. Lower is better.\n"
                "- **1px / 3px / 5px**: the fraction of valid pixels whose error is below that many pixels. Higher is better.\n"
                "- **Angular error**: the mean angle in degrees between the space-time vectors `(u, v, 1)` of the prediction and the truth.\n\n"
                "Every score is printed next to the **zero-flow baseline**, which predicts no motion anywhere. Its EPE equals the mean true "
                "displacement, so a model is only useful where it beats that row. One pair gives the verdict `sample-sanity`, not a benchmark.\n\n"
                "**Predict before running:** the background moves by about 7 px on average. What end-point error will zero flow score, and will RAFT be closer to 0.1 px or to 1 px?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "OUTPUTS = Path('outputs')\n\n"
                'NUM_FLOW_UPDATES = 12  # @param {{type:"integer"}}\n'
                'DEMO_SEED = 3  # @param {{type:"integer"}}\n\n'
                "print({{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PIXELS': MAX_PIXELS,\n"
                "       'DEFAULT_FLOW_UPDATES': DEFAULT_FLOW_UPDATES, 'MAX_FLOW': MAX_FLOW}})\n\n"
                "demo = moving_shapes_pair(DEMO_SEED)\n"
                "buffer = io.BytesIO()\n"
                "demo['image1'].save(buffer, format='PNG')\n"
                "print({{'sample_kind': 'synthetic', 'size': list(demo['image1'].size), 'sha256': hashlib.sha256(buffer.getvalue()).hexdigest()[:16],\n"
                "       'background_shift': demo['background_shift'], 'objects': [(o['kind'], o['shift']) for o in demo['objects']],\n"
                "       'valid_fraction': round(float(demo['valid'].mean()), 3)}})\n\n"
                "input_manifest = validate_inputs(demo['image1'], demo['image2'], num_flow_updates=NUM_FLOW_UPDATES)\n"
                "try:\n"
                "    validate_inputs(demo['image1'], demo['image2'].resize((128, 128)))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'probe': 'frames of different sizes', 'rejected': str(exc)}})\n"
                "print({{'verdict': input_manifest['verdict'], 'padding': input_manifest['padding'], 'findings': input_manifest['findings']}})\n\n"
                "demo_result = pipe.estimate(demo['image1'], demo['image2'], num_flow_updates=NUM_FLOW_UPDATES)\n"
                "demo_report = evaluation_report(demo_result, demo['flow'], demo['valid'], sample_kind='synthetic')\n"
                "zero_row = demo_report['baselines'][0]\n"
                "print(f\"{{'':<10s}} {{'EPE':>8s}} {{'1px':>7s}} {{'3px':>7s}} {{'5px':>7s}} {{'angle':>8s}}\")\n"
                "for name, row in (('RAFT', demo_report['metrics']), ('zero-flow', zero_row)):\n"
                "    print(f\"{{name:<10s}} {{row['epe']:>8.3f}} {{row['1px']:>7.3f}} {{row['3px']:>7.3f}} {{row['5px']:>7.3f}} {{row['angular_error_deg']:>8.2f}}\")\n"
                "print({{'verdict': demo_report['verdict'], 'valid_pixels': demo_report['metrics']['valid_pixels']}})\n\n"
                "strip = Image.new('RGB', (4 * 256, 256))\n"
                "for index, panel in enumerate((demo['image1'], demo['image2'], flow_to_rgb(demo['flow']), flow_to_rgb(demo_result['flow']))):\n"
                "    strip.paste(panel.resize((256, 256)), (256 * index, 0))\n"
                "print('frame 1 | frame 2 | true flow | predicted flow (hue = direction, saturation = speed)')\n"
                "strip"
            ),
        },
        {
            "md": (
                '**What to notice:** the RAFT row against the zero-flow row, the validity fraction, the mismatched-size rejection, and the four-panel strip.\n\n<details><summary>Check '
                'your reasoning</summary>Zero flow scores the mean true displacement; in the recorded run that was EPE 6.8937 against 0.1625 for RAFT, with 98.8 % of '
                'valid pixels within 1 px. Integer shifts of textured frames are an easy case, which is why the zero-flow row, not 0, is the reference, and why one '
                'rendered pair is `sample-sanity`, not a benchmark.</details>'
            ),
        },
        # ---------------------------------------------------------------- 5. Iterations & probes
        {
            "md": (
                "## 5. Recurrent updates and degenerate input probes\n\n"
                "RAFT's estimate depends on how many times the update block refines it. torchvision's default is 12 updates, and the builder "
                "accepts any count; this pipeline allows 1 to 32. The first table re-scores the demonstration pair at 1, 4, 12 and 32 updates, so "
                "you can see where more refinement stops paying for its compute.\n\n"
                "Then ask the model about pairs whose answer is known before trusting it:\n"
                "- **Static pair:** the same textured frame twice. The true flow is zero everywhere, so any predicted motion is error.\n"
                "- **Blank pair:** two featureless white frames. Motion is undetermined, because nothing can be matched. The cell reports the "
                "mean predicted displacement; a large value is motion the model invented.\n\n"
                "**What to look for:** EPE close to zero on the static pair, and a small mean displacement on the blank pair.\n\n"
                "**Predict before running:** will 32 updates always beat 12? And on two blank white frames, will the model report no motion at all?"
            ),
            "code": (
                "iteration_sweep = {{}}\n"
                "for updates in (1, 4, 12, 32):\n"
                "    flow = pipe.estimate(demo['image1'], demo['image2'], num_flow_updates=updates)['flow']\n"
                "    iteration_sweep[updates] = {{k: round(v, 4) for k, v in flow_metrics(flow, demo['flow'], demo['valid']).items() if k != 'valid_pixels'}}\n"
                "    print(f'{{updates:>2d}} updates', iteration_sweep[updates])\n\n"
                "degenerate = {{}}\n"
                "for probe in (static_pair(), blank_pair()):\n"
                "    flow = pipe.estimate(probe['image1'], probe['image2'], num_flow_updates=NUM_FLOW_UPDATES)['flow']\n"
                "    speed = np.linalg.norm(flow, axis=2)\n"
                "    degenerate[probe['id']] = {{'epe_vs_zero_truth': round(float(speed.mean()), 4), 'max_displacement_px': round(float(speed.max()), 4)}}\n"
                "print(json.dumps(degenerate, indent=2))"
            ),
        },
        {
            "md": (
                '**What to notice:** the EPE at 1, 4, 12 and 32 updates, and the static and blank probe values.\n\n<details><summary>Check your reasoning</summary>No and '
                'no. In the recorded run EPE fell from 0.4254 (1 update) to 0.1625 (12) but rose to 0.1893 at 32 — more refinement can drift. The static pair was near '
                'zero (EPE 0.0095), but the blank pair produced up to 0.8614 px of motion that does not exist: on featureless input the model invents flow.</details>'
            ),
        },
        # ---------------------------------------------------------------- 6. Dataset & Validation
        {
            "md": (
                "## 6. Labelled adaptation dataset and validation\n\n"
                "`flow_dataset` renders a deterministic dataset of `N_PAIRS` pairs like the demonstration pair, each from its own seed, with "
                "background displacements up to 6 px and object displacements up to 12 px in each axis. Every record carries `image1`, `image2`, "
                "the exact `flow` and the `valid` mask.\n\n"
                "`validate_dataset` checks every record before any model runs: the record keys, that both frames share one size inside the "
                "ceilings, that the flow has shape `(H, W, 2)` and is finite, and that the valid mask is boolean, has the frame's shape and selects "
                "at least one pixel. It returns a dataset manifest with the mean displacement and the mean valid fraction, and a finding for any "
                "valid pixel that moves `MAX_FLOW` (400) px or more, which the loss ignores."
            ),
            "code": (
                'N_PAIRS = 40  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n'
                'EPOCHS = 3  # @param {{type:"integer"}}\n\n'
                "records = flow_dataset(N_PAIRS, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, epochs=EPOCHS)\n"
                "print(json.dumps(dataset_manifest, indent=2))\n\n"
                "preview = Image.new('RGB', (3 * 160, 2 * 160))\n"
                "for index, record in enumerate(records[:3]):\n"
                "    preview.paste(record['image1'].resize((160, 160)), (160 * index, 0))\n"
                "    preview.paste(flow_to_rgb(record['flow']).resize((160, 160)), (160 * index, 160))\n"
                "preview"
            ),
        },
        # ---------------------------------------------------------------- 7. Split & Baseline
        {
            "md": (
                "## 7. Split and measure the pre-adaptation baseline\n\n"
                "The dataset is split at random into a training part (`1 - HOLDOUT`) and a held-out part (`HOLDOUT`); with the defaults `N_PAIRS = 40` and `HOLDOUT = 0.25` that is 30 training and 10 held-out pairs, and the cell prints the actual counts. A random split is "
                "valid here because every pair is rendered independently. Pairs cut from the same video must be split by video instead, or the "
                "held-out frames will share content with the training frames. The held-out pairs are never shown to the optimizer, and no "
                "hyperparameter is selected on them.\n\n"
                "The fine-tune runs on its own copy of the verified model, `adapter`, so `pipe` stays the pretrained reference. The baseline is "
                "that copy's held-out score before any update.\n\n"
                "**The pretrained model already beats the zero-flow baseline here.** Textured frames moved by whole pixels are an easy case for "
                "RAFT, so a large gap between the two baseline rows is expected. The fine-tune has to improve on the RAFT row, not on zero flow.\n\n"
                "*Evaluation practice.* **Predict before running:** why is a random split acceptable for these rendered pairs, when it would not be for frames cut from one video?"
            ),
            "code": (
                'HOLDOUT = 0.25  # @param {{type:"number"}}\n'
                'SEED = 0  # @param {{type:"integer"}}\n\n'
                "train_records, held_out = split_pairs(records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "overlap = {{r['id'] for r in train_records}} & {{r['id'] for r in held_out}}\n"
                "assert not overlap, f'split leaked records: {{sorted(overlap)}}'\n"
                "print({{'train': len(train_records), 'held_out': len(held_out)}})\n\n"
                "adapter = RaftPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "baseline = adapter.evaluate(held_out, num_flow_updates=NUM_FLOW_UPDATES)\n"
                "METRIC_KEYS = ('epe', '1px', '3px', '5px', 'angular_error_deg')\n"
                "print(json.dumps({{'pretrained': {{k: round(baseline[k], 4) for k in METRIC_KEYS}},\n"
                "                  'zero_flow': {{k: round(baseline['zero_flow_baseline'][k], 4) for k in METRIC_KEYS}}}}, indent=2))"
            ),
        },
        {
            "md": (
                '**What to notice:** the printed train / held-out counts (30 / 10 with the defaults), the pretrained row against the zero-flow row on the held-out pairs.\n\n<details><summary>Check your '
                'reasoning</summary>Every rendered pair comes from its own seed, so no two pairs share content and a random split cannot leak. Frames from one video '
                'share scenes, so a random split there would test on near-copies of training frames. In the recorded run the held-out EPE was 0.1649 pretrained against '
                '4.9648 for zero flow.</details>'
            ),
        },
        # ---------------------------------------------------------------- 8. Bounded Fine-Tuning
        {
            "md": (
                "## 8. Bounded fine-tuning\n\n"
                "This cell runs the real adaptation step in this runtime. It is gradient fine-tuning with RAFT's **sequence loss**, the loss of "
                "the RAFT paper and of torchvision's reference training script. The model returns one flow estimate per recurrent update; the "
                "loss is the L1 distance to the true flow over valid pixels for each estimate, weighted by `gamma ** (N - i - 1)` with `gamma = 0.8`, "
                "so the last estimate weighs most. Pixels whose true displacement is 400 px or more are ignored, as in the reference.\n\n"
                "- **The feature and context encoders are frozen by default.** With `FREEZE_ENCODERS = True` their weights keep the pretrained "
                "values, and only the recurrent update block and the upsampling mask predictor train. The cell prints the trainable and total "
                "parameter counts.\n"
                "- **Every BatchNorm layer is held in evaluation mode**, so its running statistics do not drift either, as torchvision's reference "
                "does in its Sintel stage (`--freeze-batch-norm`).\n"
                "- **Schedule:** `EPOCHS` epochs of AdamW at learning rate `2e-5` and weight decay `5e-5`, batch size 2, 12 recurrent updates, "
                "gradients clipped to norm 1.0, float32, seed `SEED`.\n\n"
                "**Read the loss as optimisation evidence only.** A falling loss says the optimizer is fitting the training pairs; the held-out "
                "end-point error in the next section is the task evidence.\n\n"
                "**Re-running this cell is safe.** It rebuilds `adapter` from the verified checkpoint before training, so each run is one fine-tune "
                "from the pretrained weights, never a second one stacked on already-adapted weights; the pretrained column in Section 9 still comes "
                "from Section 7, which scored an untouched copy.\n\n"
                "**Predict before running:** with the encoders frozen, how many of the model's parameters will train?"
            ),
            "code": (
                'LEARNING_RATE = 2e-5  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 2  # @param {{type:"integer"}}\n'
                'FREEZE_ENCODERS = True  # @param {{type:"boolean"}}\n\n'
                "# Start from the verified checkpoint on every run of this cell (RF-M3), so changing a field and re-running it repeats\n"
                "# one fine-tune instead of continuing from the weights the previous run adapted.\n"
                "adapter = RaftPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "print({{'adapter': 'rebuilt from the verified checkpoint', 'adapted': adapter.adapted}})\n"
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_encoders=FREEZE_ENCODERS,\n"
                "    progress=lambda row: print(f\"epoch {{row['epoch']}}/{{row['epochs']}}  loss {{row['loss']:.4f}}\"),\n"
                ")\n"
                "print(json.dumps({{key: run[key] for key in ('freeze_encoders', 'batchnorm', 'trainable_parameters', 'total_parameters', 'epochs',\n"
                "                                         'batch_size', 'learning_rate', 'optimizer', 'loss', 'precision', 'device', 'epoch_losses')}}, indent=2))"
            ),
        },
        {
            "md": (
                '**What to notice:** the trainable and total parameter counts, `freeze_encoders`, `batchnorm`, and the per-epoch losses.\n\n<details><summary>Check your '
                'reasoning</summary>About three in five. In the recorded run 3,120,960 of 5,257,536 parameters trained; the loss fell 0.4394 → 0.3890 → 0.3614 over '
                'three epochs. A falling training loss is optimisation evidence only; Section 9 is the task evidence.</details>'
            ),
        },
        # ---------------------------------------------------------------- 9. Evaluate Held-Out
        {
            "md": (
                "## 9. Evaluate on the held-out split\n\n"
                "`evaluate` re-runs on the same held-out pairs with the same number of updates as the baseline, so the rows are comparable. Each "
                "metric is the mean over pairs, each pair weighing the same; `pixel_weighted_epe` weighs every valid pixel the same instead. These "
                "are tutorial metrics from one pass over the held-out synthetic pairs (10 with the defaults), with no dispersion estimate. This "
                "notebook does not measure how far a different `SEED` moves these numbers, so a change of a few hundredths of a pixel may be "
                "within seed variation.\n\n"
                "*Evaluation practice.* **Predict before running:** the pretrained held-out EPE is already about 0.16 px. Will the fine-tune lower it on every held-out pair?"
            ),
            "code": (
                "adapted = adapter.evaluate(held_out, num_flow_updates=NUM_FLOW_UPDATES)\n"
                "print(f\"{{'metric':<18s}} {{'zero-flow':>10s}} {{'pretrained':>11s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "for key in METRIC_KEYS + ('pixel_weighted_epe',):\n"
                "    zero = baseline['zero_flow_baseline'][key]\n"
                '    print(f"{{key:<18s}} {{zero:>10.4f}} {{baseline[key]:>11.4f}} {{adapted[key]:>10.4f}} {{adapted[key] - baseline[key]:>+10.4f}}")'
            ),
        },
        {
            "md": (
                '**What to notice:** the change column for every metric, and `pixel_weighted_epe` beside the per-pair mean.\n\n<details><summary>Check your '
                'reasoning</summary>In the recorded run, yes: held-out EPE went 0.1649 → 0.1446 and was lower on all 10 pairs (1 px rate 0.9855 → 0.9878). Ten rendered '
                'pairs and one seed carry no dispersion estimate, and the model was fitted to renderings from the same generator, so this says nothing about camera '
                'footage.</details>\n\n**Checkpoint: is a gain of about 0.02 px real?** Before opening the answer, decide what you would need to see to '
                'believe it.\n\n<details><summary>Check your reasoning</summary>Not established by this run. The gain (0.1649 → 0.1446, −0.020 px) is consistent '
                '— every held-out pair improved in the recorded run — but the notebook never measures how far a different `SEED` moves the scores, so the gain is '
                'stated against an *unmeasured* variation, and in Section 10 one of three unseen pairs got slightly worse (0.0976 → 0.0987). To measure it, set '
                '`SEED` to 1 and then 2 in Section 7, rerun Sections 7–9 each time, and compare the spread of the adapted EPE with the gain.</details>'
            ),
        },
        {
            "md": (
                "### Activity: does unfreezing the encoders help? (Predict → Change → Run → Observe → Explain)\n\n"
                "1. **Predict.** Write down whether held-out EPE will fall further, stay or rise when the feature and context encoders train too, and "
                "how `trainable_parameters` will change.\n"
                "2. **Change.** In Section 8 set `FREEZE_ENCODERS = False`.\n"
                "3. **Run.** Run the Section 8 cell, then the Section 9 cell. Section 8 rebuilds `adapter` from the verified checkpoint every time, so "
                "this is one fresh fine-tune with the new setting, not a second one stacked on the first. (To change `EPOCHS` or `N_PAIRS`, rerun "
                "from Section 6; to change `NUM_FLOW_UPDATES`, rerun from Section 4.)\n"
                "4. **Observe.** Compare `trainable_parameters`, `epoch_losses` and the Section 9 change column with your first run.\n"
                "5. **Explain.** Is the difference larger than the seed variation the Section 9 checkpoint asks about? Then set "
                "`FREEZE_ENCODERS = True` again and rerun Sections 8–12, so the exported adapter and the output files describe the default run.\n\n"
                "<details><summary>Check your reasoning</summary>Unfrozen, all 5,257,536 parameters train instead of 3,120,960. In the review's local "
                "CPU run of this exercise (4 October 2026; that CPU run reproduced the recorded T4 frozen numbers to four decimals) held-out EPE "
                "fell to 0.1089 against 0.1446 frozen. The encoders were also fitted to these renderings, so a larger gain here is still a gain on "
                "the same synthetic generator, and the comparison carries no dispersion estimate. Your GPU numbers will differ in the last "
                "digits.</details>"
            ),
        },
        # ---------------------------------------------------------------- 10. New-data inference
        {
            "md": (
                "## 10. Inference on unseen pairs\n\n"
                "Three new pairs come from a seed the dataset never used (`NEW_DATA_SEED = 99`). The pretrained and the adapted pipelines estimate "
                "flow on each, and both are scored against the exact flow.\n\n"
                "**Predict before running:** will the adapted model be better on all three unseen pairs?"
            ),
            "code": (
                'NEW_DATA_SEED = 99  # @param {{type:"integer"}}\n\n'
                "new_records = flow_dataset(3, seed=NEW_DATA_SEED)\n"
                "new_data_rows = []\n"
                "for record in new_records:\n"
                "    row = {{'id': record['id'], 'mean_true_displacement_px': round(float(np.linalg.norm(record['flow'], axis=2)[record['valid']].mean()), 3)}}\n"
                "    for name, model in (('pretrained', pipe), ('adapted', adapter)):\n"
                "        flow = model.estimate(record['image1'], record['image2'], num_flow_updates=NUM_FLOW_UPDATES)['flow']\n"
                "        row[f'{{name}}_epe'] = round(flow_metrics(flow, record['flow'], record['valid'])['epe'], 4)\n"
                "    new_data_rows.append(row)\n"
                "    print(json.dumps(row))"
            ),
        },
        {
            "md": (
                '**What to notice:** pretrained against adapted EPE for each unseen pair.\n\n<details><summary>Check your reasoning</summary>No. In the recorded run two '
                'pairs improved (0.0978 → 0.0921, 0.2836 → 0.2161) and one got slightly worse (0.0976 → 0.0987). A held-out mean can improve while individual cases '
                'regress; report both.</details>'
            ),
        },
        # ---------------------------------------------------------------- 11. Export, reload & verify
        {
            "md": (
                "## 11. Adapter export, fresh reload, and equivalence check\n\n"
                "`save_artifact` writes `outputs/raft_adapter.safetensors`: every tensor the fine-tune could change, plus a metadata header naming "
                "the base model, its pinned checkpoint SHA-256 and the frozen prefixes. With the encoders frozen, their tensors are left out because "
                "they equal the verified base checkpoint.\n\n"
                "`load_artifact` then builds a **fresh** pipeline from the verified base checkpoint, loads the adapter tensors onto it, and refuses "
                "an adapter whose format, base identity, base digest or tensor set does not fit. The cell compares the reloaded flow with the "
                "in-memory model's on an unseen pair, with a stated tolerance: loading succeeding is not the check, reproducing the flow is."
            ),
            "code": (
                "artifact_path = OUTPUTS / 'raft_adapter.safetensors'\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='RAFT-Large moving-shapes adaptation tutorial adapter')\n"
                "print(json.dumps(descriptor, indent=2))\n\n"
                "reloaded = RaftPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)\n"
                "print({{'reloaded_source': reloaded.source, 'adapted': reloaded.adapted, 'frozen_prefixes': list(reloaded.frozen_prefixes)}})\n\n"
                "TOLERANCE_PX = 1e-3\n"
                "probe = new_records[0]\n"
                "flow_orig = adapter.estimate(probe['image1'], probe['image2'], num_flow_updates=NUM_FLOW_UPDATES)['flow']\n"
                "flow_reloaded = reloaded.estimate(probe['image1'], probe['image2'], num_flow_updates=NUM_FLOW_UPDATES)['flow']\n"
                "difference = float(np.linalg.norm(flow_orig - flow_reloaded, axis=2).mean())\n"
                "assert difference <= TOLERANCE_PX, f'reloaded flow differs by {{difference:.6f}} px on average'\n"
                "reload_check = {{'pair': probe['id'], 'mean_endpoint_difference_px': difference, 'tolerance_px': TOLERANCE_PX, 'equivalent': True}}\n"
                "print(reload_check)"
            ),
        },
        {
            "md": (
                '**What to notice:** the adapter descriptor (tensor count, size, base digest) and `mean_endpoint_difference_px` against the tolerance.\n\n<details><summary>Check '
                'your reasoning</summary>In the recorded run the 30-tensor, 12.5 MB adapter reloaded onto a fresh verified base and reproduced the flow exactly (mean '
                'difference 0.0 px against a 0.001 px tolerance). Loading is not the check; reproducing the flow is.</details>'
            ),
        },
        # ---------------------------------------------------------------- 12. Outputs & provenance
        {
            "md": (
                "## 12. Write machine-readable outputs and provenance\n\n"
                "The cell writes:\n"
                "- `outputs/raft_optical_flow_input_manifest.json`\n"
                "- `outputs/raft_optical_flow_evaluation_report.json`\n"
                "- `outputs/raft_optical_flow_result.json` (identity, runtime versions, device, iteration sweep, probes, dataset manifest, split, "
                "baseline and adapted metrics, fine-tuning configuration, new-data rows, adapter descriptor and reload check)\n"
                "- `outputs/raft_optical_flow_metrics.csv` (one row per held-out and unseen pair, pretrained and adapted EPE)\n"
                "- `outputs/raft_optical_flow_flow.flo` and `outputs/raft_optical_flow_flow.npy` (the demonstration pair's predicted flow)\n"
                "- `outputs/raft_optical_flow_flow.png` (the same flow, colour-coded)\n"
                "- `outputs/raft_adapter.safetensors` (written in Section 11)"
            ),
            "code": (
                "import csv\n\n"
                "with open(OUTPUTS / '{stem}_input_manifest.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(input_manifest, f, indent=2)\n\n"
                "with open(OUTPUTS / '{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(demo_report, f, indent=2)\n\n"
                "write_flo(OUTPUTS / '{stem}_flow.flo', demo_result['flow'])\n"
                "np.save(OUTPUTS / '{stem}_flow.npy', demo_result['flow'])\n"
                "flow_to_rgb(demo_result['flow']).save(OUTPUTS / '{stem}_flow.png')\n"
                "assert np.array_equal(read_flo(OUTPUTS / '{stem}_flow.flo'), demo_result['flow'])\n\n"
                "with open(OUTPUTS / '{stem}_metrics.csv', 'w', newline='', encoding='utf-8') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['split', 'pair', 'pretrained_epe', 'adapted_epe', 'zero_flow_epe'])\n"
                "    for split, group in (('held_out', held_out), ('unseen', new_records)):\n"
                "        for record in group:\n"
                "            scores = [flow_metrics(m.estimate(record['image1'], record['image2'], num_flow_updates=NUM_FLOW_UPDATES)['flow'], record['flow'], record['valid'])['epe'] for m in (pipe, adapter)]\n"
                "            zero = flow_metrics(np.zeros_like(record['flow']), record['flow'], record['valid'])['epe']\n"
                "            writer.writerow([split, record['id'], *(f'{{v:.4f}}' for v in (*scores, zero))])\n\n"
                "result_export = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'weights_url': WEIGHTS_URL,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,\n"
                "                'cuda': torch.cuda.is_available()}},\n"
                "    'device': pipe.device,\n"
                "    'num_flow_updates': NUM_FLOW_UPDATES,\n"
                "    'demo_metrics': demo_report['metrics'],\n"
                "    'iteration_sweep': iteration_sweep,\n"
                "    'degenerate_probes': degenerate,\n"
                "    'adaptation': {{\n"
                "        'dataset': dataset_manifest,\n"
                "        'dataset_seed': DATASET_SEED,\n"
                "        'split': {{'train': len(train_records), 'held_out': len(held_out), 'seed': SEED, 'holdout': HOLDOUT}},\n"
                "        'finetune': run,\n"
                "        'baseline': baseline,\n"
                "        'adapted': adapted,\n"
                "        'new_data': new_data_rows,\n"
                "        'artifact': descriptor,\n"
                "        'reload_check': reload_check,\n"
                "    }},\n"
                "}}\n"
                "with open(OUTPUTS / '{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_export, f, indent=2, default=str)\n\n"
                "for p in sorted(OUTPUTS.iterdir()):\n"
                "    if p.is_file():\n"
                "        print(f'  {{p.name:<44s}} {{p.stat().st_size:>12,d}} bytes')"
            ),
        },
        # ---------------------------------------------------------------- 13. BYOD
        {
            "md": (
                "## 13. Optional: Bring Your Own Data (BYOD)\n\n"
                "Both branches are off by default, so `Run all` never stops here. Before you turn one on, read the contract:\n\n"
                "- **Frame-pair branch** (`USE_BYOD_IMAGE`): two image files PIL can open, of the same size, each side between `MIN_IMAGE_SIDE` "
                "(128) and `MAX_IMAGE_SIDE` (1024) px and at most `MAX_PIXELS` (786,432) pixels. Frame 1 comes before frame 2 in time. The pair "
                "runs through `validate_inputs`, `estimate` and `evaluation_report`; the report is `not-measurable`, because no ground-truth flow "
                "comes with it.\n"
                "- **Dataset branch** (`USE_BYOD_DATASET`): a directory with `pairs.json` (a list of `{{'frame1', 'frame2', 'flow', 'valid'}}` "
                "objects) and the files it names, at least 2 records and at most 2,000. `flow` is a Middlebury `.flo` file or a `.npy` array of "
                "shape `(H, W, 2)`; `valid` is an optional single-channel PNG whose non-zero pixels are valid. File names must stay inside the "
                "directory. The branch runs the same validate → split → baseline → fine-tune → evaluate → export → reload stages as the sample, "
                "checks that the reloaded adapter reproduces the flow on a held-out pair, and writes the zero-flow, pretrained and adapted metrics, "
                "the split sizes and the reload check to `outputs/byod_result.json`. With fewer than 5 held-out pairs it prints a small-split "
                "warning: the documented minimum of 2 records leaves 1 held-out pair, which is an anecdote, not an estimate.\n\n"
                "Set `BYOD_IMAGE1_PATH` and `BYOD_IMAGE2_PATH`, or `BYOD_DATASET_DIR`, to read from a mounted or local location; leave them empty "
                "on Colab to get an upload dialog instead (for the pair branch, upload both frames at once; they are ordered by file name). "
                "Uploaded files are written under `outputs/byod/` in this runtime and are not sent anywhere else. The first lines of the cell show "
                "the validator refusing two malformed inputs with messages that name the failed rule."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                'BYOD_IMAGE1_PATH = ""  # @param {{type:"string"}}\n'
                'BYOD_IMAGE2_PATH = ""  # @param {{type:"string"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                'BYOD_DATASET_DIR = ""  # @param {{type:"string"}}\n\n'
                "_probe = static_pair()\n"
                "for desc, probe_call in (\n"
                "    ('non-image object', lambda: validate_inputs('/not/an/image.png', _probe['image2'])),\n"
                "    ('flow of the wrong shape', lambda: validate_dataset([{{'image1': _probe['image1'], 'image2': _probe['image2'],\n"
                "                                                          'flow': np.zeros((8, 8, 2), np.float32)}}])),\n"
                "):\n"
                "    try:\n"
                "        probe_call()\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print(f'refused as expected: {{desc}} -> {{type(exc).__name__}}: {{exc}}')\n\n"
                "BYOD_DIR = OUTPUTS / 'byod'\n"
                "BYOD_MIN_HELD_OUT = 5  # fewer held-out pairs than this prints a small-split warning\n\n"
                "def _upload_into(target, fields):\n"
                "    try:\n"
                "        from google.colab import files  # type: ignore[import-not-found]\n"
                "    except ImportError:\n"
                "        raise ValueError(f'The upload dialog exists only on Google Colab. Set {{fields}} to a local or mounted path '\n"
                "                         'and run this cell again.') from None\n"
                "    target.mkdir(parents=True, exist_ok=True)\n"
                "    for name, data in files.upload().items():\n"
                "        (target / Path(name).name).write_bytes(data)\n"
                "    return target\n\n"
                "if USE_BYOD_IMAGE:\n"
                "    if BYOD_IMAGE1_PATH and BYOD_IMAGE2_PATH:\n"
                "        frame_paths = [Path(BYOD_IMAGE1_PATH), Path(BYOD_IMAGE2_PATH)]\n"
                "    else:\n"
                "        frame_paths = sorted(p for p in _upload_into(BYOD_DIR / 'pair', 'BYOD_IMAGE1_PATH and BYOD_IMAGE2_PATH').iterdir() if p.is_file())[:2]\n"
                "    if len(frame_paths) != 2:\n"
                "        raise ValueError(f'the frame-pair branch needs exactly two frames, got {{len(frame_paths)}}')\n"
                "    byod_frames = []\n"
                "    for frame_path in frame_paths:\n"
                "        with Image.open(frame_path) as handle:\n"
                "            byod_frames.append(handle.convert('RGB'))\n"
                "    print(validate_inputs(*byod_frames, num_flow_updates=NUM_FLOW_UPDATES)['verdict'], [p.name for p in frame_paths])\n"
                "    byod_result = pipe.estimate(*byod_frames, num_flow_updates=NUM_FLOW_UPDATES)\n"
                "    byod_report = evaluation_report(byod_result, sample_kind='byod')\n"
                "    print({{'verdict': byod_report['verdict'], 'mean_predicted_displacement_px': round(byod_report['mean_predicted_displacement_px'], 3)}})\n"
                "    write_flo(OUTPUTS / 'byod_flow.flo', byod_result['flow'])\n"
                "    flow_to_rgb(byod_result['flow']).save(OUTPUTS / 'byod_flow.png')\n"
                "else:\n"
                "    print('BYOD frame-pair branch is off; set USE_BYOD_IMAGE = True to estimate flow between your own two frames.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    dataset_dir = Path(BYOD_DATASET_DIR) if BYOD_DATASET_DIR else _upload_into(BYOD_DIR / 'dataset', 'BYOD_DATASET_DIR')\n"
                "    byod_records = read_flow_records(dataset_dir)\n"
                "    byod_manifest = validate_dataset(byod_records, epochs=EPOCHS)\n"
                "    print(json.dumps(byod_manifest, indent=2))\n"
                "    byod_train, byod_held = split_pairs(byod_records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "    print({{'train': len(byod_train), 'held_out': len(byod_held)}})\n"
                "    if len(byod_held) < BYOD_MIN_HELD_OUT:\n"
                "        print(f'WARNING: only {{len(byod_held)}} held-out pair(s). The scores below are anecdotes, not an estimate; '\n"
                "              f'supply enough pairs for at least {{BYOD_MIN_HELD_OUT}} held-out pairs before drawing a conclusion.')\n"
                "    byod_pipe = RaftPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "    byod_baseline = byod_pipe.evaluate(byod_held, num_flow_updates=NUM_FLOW_UPDATES)\n"
                "    byod_run = byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, freeze_encoders=FREEZE_ENCODERS)\n"
                "    byod_adapted = byod_pipe.evaluate(byod_held, num_flow_updates=NUM_FLOW_UPDATES)\n"
                "    print(f\"{{'metric':<18s}} {{'zero-flow':>10s}} {{'pretrained':>11s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "    for key in METRIC_KEYS + ('pixel_weighted_epe',):\n"
                "        zero = byod_baseline['zero_flow_baseline'][key]\n"
                "        print(f\"{{key:<18s}} {{zero:>10.4f}} {{byod_baseline[key]:>11.4f}} {{byod_adapted[key]:>10.4f}} {{byod_adapted[key] - byod_baseline[key]:>+10.4f}}\")\n"
                "    byod_artifact = OUTPUTS / 'byod_raft_adapter.safetensors'\n"
                "    byod_descriptor = byod_pipe.save_artifact(byod_artifact, notes='BYOD adaptation adapter')\n"
                "    byod_reloaded = RaftPipeline.load_artifact(byod_artifact, weights_dir=WEIGHTS_DIR)\n"
                "    byod_probe = byod_held[0]\n"
                "    byod_flows = [m.estimate(byod_probe['image1'], byod_probe['image2'], num_flow_updates=NUM_FLOW_UPDATES)['flow'] for m in (byod_pipe, byod_reloaded)]\n"
                "    byod_difference = float(np.linalg.norm(byod_flows[0] - byod_flows[1], axis=2).mean())\n"
                "    assert byod_difference <= TOLERANCE_PX, f'reloaded BYOD flow differs by {{byod_difference:.6f}} px on average'\n"
                "    byod_reload_check = {{'pair': byod_probe['id'], 'mean_endpoint_difference_px': byod_difference, 'tolerance_px': TOLERANCE_PX, 'equivalent': True}}\n"
                "    print(byod_reload_check)\n"
                "    byod_export = {{\n"
                "        'notebook_source': NOTEBOOK_SOURCE, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'sample_kind': 'byod',\n"
                "        'dataset': byod_manifest,\n"
                "        'split': {{'train': len(byod_train), 'held_out': len(byod_held), 'seed': SEED, 'holdout': HOLDOUT,\n"
                "                  'small_split_warning': len(byod_held) < BYOD_MIN_HELD_OUT}},\n"
                "        'finetune': byod_run, 'baseline': byod_baseline, 'adapted': byod_adapted,\n"
                "        'artifact': byod_descriptor, 'reload_check': byod_reload_check,\n"
                "    }}\n"
                "    with open(OUTPUTS / 'byod_result.json', 'w', encoding='utf-8') as f:\n"
                "        json.dump(byod_export, f, indent=2, default=str)\n"
                "    print('BYOD adapter exported, reloaded and checked; results in outputs/byod_result.json:', byod_descriptor['sha256'][:16])\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt RAFT on your own frame pairs with ground-truth flow.')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**What this notebook established, in this runtime.** The pinned torchvision checkpoint for `torchvision/raft_large` was verified "
        "against a committed SHA-256 manifest before loading. The pretrained model estimated flow on a rendered pair with exact ground truth, "
        "was re-scored at four update counts, and was probed with a static and a featureless pair. A labelled dataset of rendered pairs was "
        "then used to fine-tune the recurrent update block with RAFT's sequence loss, with the encoders and every BatchNorm layer frozen, and "
        "the held-out split was scored by end-point error before and after, next to the zero-flow baseline. The adapter was exported, reloaded "
        "onto a fresh base model and checked against the in-memory flow.\n\n"
        "**How large the gain is.** With the defaults the recorded run moved held-out EPE from 0.1649 to 0.1446 px (−0.020 px), lower on every "
        "held-out pair, while one of three unseen pairs got slightly worse (0.0976 → 0.0987). Seed variation was not measured, so this notebook "
        "cannot say whether a gain of this size exceeds what a different `SEED` would produce; Section 9's checkpoint shows how to measure it.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set and the "
        "pinned checkpoint together reproduce these stages in a fresh runtime, without the repository being cloned or installed and without "
        "any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the adapted model "
        "generalises beyond the rendered pairs it was fitted to. The held-out EPE is measured on 10 rendered pairs with the default `N_PAIRS` and `HOLDOUT` and carries no "
        "dispersion estimate.\n\n"
        "**Reproducibility.** Seeds are form fields (`DEMO_SEED`, `DATASET_SEED`, `SEED`, `NEW_DATA_SEED`), and the run is float32 with no data "
        "augmentation. GPU kernels (the correlation lookup's `grid_sample` backward pass in particular) are not forced to be deterministic, so "
        "repeated GPU runs can differ in the last digits of the loss and the scores.\n\n"
        "**Try next.** The activity after Section 9 compares `FREEZE_ENCODERS = False` with the default. To change `NUM_FLOW_UPDATES` "
        "and watch the EPE and the time per pair, set it in Section 4 and rerun from Section 4. To transfer the workflow, point `BYOD_DATASET_DIR` at frame pairs with ground-truth flow from your own "
        "domain, for example rendered from a simulator.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 3 reports a size or SHA-256 mismatch for the '
        'checkpoint** — the download from `download.pytorch.org` was cut short or altered; delete the file in the snapshot folder and run Section 3 again.\n- '
        '**The fine-tune is slow** — you are on CPU; switch to a T4 GPU.\n- **Your numbers differ in the last digits from the recorded run** — GPU kernels (the '
        "correlation lookup's backward pass) are not deterministic; the comparison between rows is the result.\n- **BYOD: a frame-size, `pairs.json` or "
        'flow-shape refusal** — the message names the rule; frames must share one size inside the ceilings, and flow must be a `.flo` or `(H, W, 2)` `.npy` '
        'file. Set `BYOD_IMAGE1_PATH`/`BYOD_IMAGE2_PATH` or `BYOD_DATASET_DIR` to skip the upload dialog.\n\n## Glossary\n\n- **Optical flow:** the per-pixel '
        'displacement `(u, v)` that maps each point of frame 1 to where it appears in frame 2.\n- **Valid mask:** the pixels whose destination is visible in '
        'frame 2; only these are scored.\n- **EPE (end-point error):** the mean distance in pixels between predicted and true displacement; lower is better.\n- '
        '**1px / 3px / 5px:** the share of valid pixels whose error is under that many pixels.\n- **Angular error:** the mean angle between the space-time '
        'vectors `(u, v, 1)` of prediction and truth.\n- **Zero-flow baseline:** predicting no motion anywhere; its EPE equals the mean true displacement.\n- '
        "**Recurrent updates:** RAFT refines its estimate a fixed number of times; 12 by default.\n- **Sequence loss:** RAFT's training loss: the L1 error of "
        'every intermediate estimate, later ones weighted more.\n- **Frozen encoders / BatchNorm:** the feature and context encoders keep their pretrained '
        'weights, and normalisation statistics do not drift.\n- **Adapter:** the SafeTensors file holding only the tensors the fine-tune changed, reloaded onto '
        'the verified base.\n- **Isolated environment:** the separate Python environment Section 1 builds from the hash lock; every later cell runs there.\n\n## '
        "Conclusion (your notes)\n\nComplete these in your own words; the recorded run's values are in the **Check your reasoning** answers above.\n\n- On the "
        "demonstration pair RAFT scored EPE ___ against zero flow's ___; the blank-pair probe showed ___.\n- The fine-tune moved held-out EPE from ___ to ___; "
        'on unseen pairs ___.\n- The number I would not trust on its own is ___, because ___.\n- Before adapting on my own footage I would split by ___ and '
        'compare against ___.\n\n'
        "## References\n\n"
        "- Teed, Z. and Deng, J. (2020). *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow.* ECCV 2020. [arXiv:2003.12039](https://arxiv.org/abs/2003.12039).\n"
        "- torchvision model documentation: [raft_large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html); upstream repository [pytorch/vision](https://github.com/pytorch/vision) — BSD-3-Clause; training recipe [references/optical_flow](https://github.com/pytorch/vision/tree/main/references/optical_flow).\n"
        "- Butler, D. J., Wulff, J., Stanley, G. B. and Black, M. J. (2012). *A Naturalistic Open Source Movie for Optical Flow Evaluation.* ECCV 2012 — the Sintel benchmark.\n"
        "- Baker, S., Scharstein, D., Lewis, J. P., Roth, S., Black, M. J. and Szeliski, R. (2011). *A Database and Evaluation Methodology for Optical Flow.* IJCV 92 — the Middlebury `.flo` format and the angular error.\n"
        "- Repository model card: https://github.com/kurtvalcorza/raft-optical-flow-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/raft-optical-flow-pipeline`](https://github.com/kurtvalcorza/raft-optical-flow-pipeline) — source repository for this pipeline."
    ),
}
