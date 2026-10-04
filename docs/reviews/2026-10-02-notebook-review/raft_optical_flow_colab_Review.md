# RAFT-Large Optical Flow Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch 2026-10-02)  
**Repository:** `kurtvalcorza/raft-optical-flow-pipeline`  
**Notebook:** `tutorials/raft_optical_flow_colab.ipynb`  
**Reviewed commit:** `9e8462e88cd31528dbf37a860eab49472b234cf7` (`origin/main`)  
**Notebook Git blob:** `96727669c7a6e75f6643027cc4a80e10107dfc10`  
**Framework:** Notebook Review Framework v1 · **Requirements baseline:** NOTEBOOK_SPEC 2.2 (ml-worker `origin/main`); the notebook declares 2.1  
**Finding prefix:** `RF`

## Executive assessment

The technical core is sound. The pinned checkpoint is digest-verified before loading, the default path really fine-tunes RAFT with its sequence loss on a separate `adapter` copy while `pipe` stays the pretrained reference, every score is printed next to the zero-flow baseline, the adapter reload is checked by reproduced flow rather than by "load succeeded", and the BYOD branches reach validate → split → baseline → fine-tune → evaluate → export → reload on representative input. A CPU re-execution in this review reproduced the recorded Kaggle T4 numbers to four decimals (demo EPE 0.1625; held-out 0.1649 → 0.1446).

What keeps it from release: Run all cannot finish in one pass on the documented runtime (the recorded run stopped after the install cell and needed a manual restart); the notebook declares `GUIDED` but has almost none of the guided layer; the only learner activity ("Try next: change `FREEZE_ENCODERS`") does not say which cells to rerun, and the natural rerun stacks a second fine-tune on the already-adapted weights; and REL12 (BYOD exercise in a hosted runtime) is still unrecorded, as the repository itself states.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Profile / mode | `E2E` / `GUIDED`; `standalone: true`; declares NOTEBOOK_SPEC 2.1 (current 2.2) |
| Stated learner | Basic Python, PIL and NumPy; images as pixel grids; flow as `(H, W, 2)` displacements; Euclidean distance (cell 1) |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CUDA GPU (T4) documented, CPU "much more slowly" |
| Promised outcomes | Pinned install; digest-verified checkpoint; flow on a rendered pair scored by EPE / 1-3-5px / angular error vs zero-flow; update-count sweep; static and blank probes; 40-pair labelled dataset; split + baseline; bounded fine-tune; held-out re-score; unseen-pair inference; adapter export, fresh reload, flow-equivalence check; 8 output files; two BYOD branches |
| Generator | `tools/build_notebook.py` + `tools/notebook_template.py` (regenerate, don't hand-edit) |
| Existing execution evidence | `docs/release-verification.md`: Kaggle T4, 2026-09-25, commit `97b4ab9` / blob `96727669c7a6` — **the same notebook blob as the reviewed revision**. Default path PASSED after one manual restart (pass 1 stopped with `RuntimeError: Core dependencies changed…`). REL12 not exercised; status `Candidate`. |

### Journeys and evidence basis

| Journey | Basis | Outcome |
|---|---|---|
| First-time learner | Source inspection | Clear technical narrative; guided scaffolding largely absent (RF-M2); install restart not signposted (RF-M1) |
| Clean default | Documented execution (Kaggle T4, same blob) + direct execution (local CPU, labelled) | Hosted: passes only after a manual restart. Local CPU (install skipped via the cell's own `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, env equal to `PINS`, CPU builds): 14/14 code cells ok, 138.8 s, all 8 outputs written, numbers match the hosted record |
| Active learning | Direct execution (local CPU) | "Try next" `FREEZE_ENCODERS=False`, rerunning cells 19+21 as a learner naturally would: runs, but continues from adapted weights (RF-M3) |
| Reuse and recovery | Direct execution (local CPU) | BYOD dataset (4 records at 130×133, and the documented minimum of 2) completes and exports/reloads; frame-pair branch with paths works; 1-record, missing-`flow` `pairs.json`, path-escape and 100-px frames are refused with rule-naming messages; upload path outside Colab fails with a raw `ModuleNotFoundError` (RF-m1). Not verified in a hosted runtime |

**Limitations.** No GPU, no Colab/Kaggle run by this review, no learner observation. Local CPU execution is not hosted verification. The probe environment was an existing venv whose versions equal the notebook `PINS` (CPU wheels); the install cell itself was not exercised locally.

## 2. Separate judgments

- **Technical correctness:** good. Verified snapshot loading, strict state-dict load, separate pretrained/adapted objects, frozen-encoder and BatchNorm hold, contained BYOD paths, exact `.flo` round-trip. Defects are the install/restart pattern (RF-M1) and the rerun state of `adapter` (RF-M3).
- **Promise fulfilment:** every listed default stage executes and is visible (direct + documented). The "Run all … installs …" promise holds only after a manual restart (RF-M1).
- **Learner experience:** strong explanations of EPE, validity mask, sequence loss and the zero-flow baseline; weak active learning and guidance (RF-M2, RF-M3), and some fixed numbers that the form fields can invalidate (RF-m2).
- **Spec conformance:** RUN10 / ENV6 (MUST) fail on the documented runtime; REL12 (MUST) not yet satisfied; GDL1–GDL15 (SHOULD) largely unmet; EXE2 (SHOULD) partly met; declares 2.1 (RF-m5).

## 3. Findings

### RF-M1 — Major: Run all needs a manual restart after the install cell

- **Cell/section:** cell 3, §1 "Install the pinned runtime".
- **Observed issue:** the cell `pip install`s the pins into the live kernel, then raises `RuntimeError('Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.')` when a preloaded distribution changed.
- **Consequence:** on the documented hosted runtimes Run all stops after the first code cell; the learner must restart and run again. The intro's Run all promise is not met in one pass.
- **Evidence:** documented execution — `docs/release-verification.md`, Kaggle T4 run of this blob: "pass 1 242.3 s stopped after the install cell with `RuntimeError: Core dependencies changed …` (cuda-bindings 12.9.4 → 13.4.3, numpy 2.0.2 → 2.5.3), kernel restarted". Source: generator `tools/build_notebook.py:64-72`.
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern instead of a new install guard: a carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that env so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (also `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`). Change in `tools/build_notebook.py` (install cell) and regenerate.
- **Acceptance check:** a fresh Colab or Kaggle T4 runtime executes the regenerated notebook top-to-bottom with Run all and no restart, recorded in `docs/release-verification.md` with blob id; no code cell raises a restart instruction.
- **Spec:** RUN10, ENV6 (MUST).

### RF-M2 — Major: Declared `GUIDED`, but the guided layer is essentially missing

- **Cell/section:** whole notebook.
- **Observed issue:** a static scan of the 17 markdown cells finds no prediction prompt, no interpretation checkpoint or sample answer, no glossary, no troubleshooting section, no roadmap / "how to use this notebook", and no infrastructure labels on the 36 KB carried-module cell (cell 7) or the manifest cell. There is one "What to look for" (cell 12) and one "Try next" line (cell 30). The final section states conclusions for the learner rather than asking them to form one.
- **Consequence:** a self-paced learner new to optical flow is shown results but not asked to predict, interpret or diagnose them; the 36 KB module cell appears to be required reading; when a hosted run fails (restart, download, memory, BYOD) there is no recovery guidance.
- **Evidence:** source inspection; `results.json` → `static.guided_layer_hits`.
- **Recommended correction:** in `tools/notebook_template.py`, add a short "How to use this notebook" + roadmap (GDL2–3), an Input → RAFT → flow contract (GDL4), label cells 3/5/7/9 **Infrastructure** (GDL11), a prediction before the update-count sweep and before the fine-tune (GDL7), "What to notice" after each principal stage (GDL8), 2–3 checkpoints with collapsible sample answers (e.g., why the blank pair produces motion; whether a 0.02 px gain is meaningful) (GDL9), one Predict → Change → Run → Observe → Explain activity (GDL10, see RF-M3), a glossary (EPE, outlier rate, valid mask, sequence loss, update block), a troubleshooting section (GDL13), and a conclusion template (GDL14).
- **Acceptance check:** each of GDL1–GDL14 can be pointed to a specific cell in the regenerated notebook, and the notebook still passes `build_notebook.py --check` and the parity tests.
- **Spec:** GDL1–GDL15 (SHOULD, 2.2 §3.5).

### RF-M3 — Major: The only exercise reruns the fine-tune on already-adapted weights

- **Cell/section:** cell 30 "Try next" → cells 17, 19, 21.
- **Observed issue:** "Change `FREEZE_ENCODERS` to `False` and compare held-out EPE and runtime" gives no rerun instruction. `FREEZE_ENCODERS` is a form field in cell 19, which calls `adapter.finetune(...)` on the object built in cell 17. Rerunning cell 19 (the cell holding the field) and then cell 21 continues from the adapted weights: 3 + 3 epochs, the second unfrozen, while the "pretrained" column still comes from cell 17. The same applies to `LEARNING_RATE`, `BATCH_SIZE` and `EPOCHS`/`NUM_FLOW_UPDATES` (cells 15, 11), whose dependent results stay stale unless 17 onward is rerun.
- **Consequence:** the comparison the learner is told to make is confounded (more total training plus a different freeze setting) without any sign of it; the exported adapter's descriptor would describe one fine-tune that was actually two.
- **Evidence:** direct execution, local CPU (`results.json` → `active`): after the default run, `adapter.adapted == True` and its update-block weights differ from `pipe`. Rerunning 19 (unfrozen) + 21 without 17: held-out EPE 0.1048 (5,257,536 trainable). Correct reset (17 → 19 → 21): 0.1089. The conclusion "unfreezing helps" happens to survive here, but the number reported is not the experiment described.
- **Recommended correction:** in `tools/notebook_template.py` either rebuild `adapter = RaftPipeline.from_pretrained(...)` at the top of the fine-tune cell (cell 19) so every rerun starts from the verified base, or state "rerun from Section 7 (cell 17) through Section 9" in the Try-next and §8 text; and make it a GDL10 activity (predict whether unfreezing changes held-out EPE, then run).
- **Acceptance check:** after a default run, changing `FREEZE_ENCODERS` and following the written instructions exactly yields the same held-out EPE as a fresh run with that setting (±GPU nondeterminism), and `run['epoch_losses']` has `EPOCHS` entries from a pretrained start.
- **Spec:** GDL10 (SHOULD); framework dimension 7 (an exercise leaves the notebook in an inconsistent state).

### RF-m1 — Minor: Upload fallback is Colab-only; on Jupyter/Kaggle it fails with a raw import error

- **Cell/section:** cell 29, `_upload_into`.
- **Observed issue:** leaving `BYOD_IMAGE*_PATH` / `BYOD_DATASET_DIR` empty imports `google.colab` unconditionally. Prerequisites name "Google Colab or Jupyter" as supported.
- **Evidence:** direct execution, local CPU: `USE_BYOD_IMAGE=True` with empty paths → `ModuleNotFoundError: No module named 'google'`. Generator `tools/notebook_template.py:452`.
- **Recommended correction:** catch `ImportError` and raise a message naming the location fields to set (or offer `ipywidgets.FileUpload`).
- **Acceptance check:** outside Colab, enabling a BYOD branch with empty location fields prints a message naming `BYOD_IMAGE1_PATH`/`BYOD_IMAGE2_PATH`/`BYOD_DATASET_DIR` instead of a traceback from `google`.
- **Spec:** EXE2 (SHOULD), DAT19.

### RF-m2 — Minor: Fixed numbers in prose that form fields or later runs contradict

- **Cell/section:** cells 1, 16, 20, 30.
- **Observed issue:** "Runtimes are not measured in this revision" (cell 1) while a Kaggle T4 run of this blob is recorded (277.4 s incl. restart; post-restart 35 s) and this review measured 138.8 s on CPU (fine-tune 95.6 s). "(75%, 30 pairs) … (25%, 10 pairs)" (cell 16), "10 synthetic pairs" (cell 20) and "10 rendered pairs" (cell 30) are literal while `N_PAIRS` and `HOLDOUT` are form fields and BYOD changes the counts.
- **Consequence:** a learner who changes `N_PAIRS`/`HOLDOUT` reads counts that no longer match the output; the runtime sentence hides the information a CPU user needs.
- **Evidence:** source inspection (`static.hardcoded_counts_in_markdown`, `static.runtime_claim`); documented and direct execution timings.
- **Recommended correction:** in `tools/notebook_template.py` (lines 100, 229, 293, 506) state defaults as defaults ("with the default `N_PAIRS = 40` and `HOLDOUT = 0.25`") and have the cells print the actual counts; quote the recorded T4 time and an approximate CPU time with their dates.
- **Acceptance check:** no markdown cell states a pair count without naming it as the default; the runtime line cites a recorded run.
- **Spec:** GDL8 (no hard-coded results that may vary).

### RF-m3 — Minor: The headline adaptation gain sits inside the notebook's own unmeasured noise band

- **Cell/section:** cells 20–21, 30.
- **Observed issue:** cell 20 says "A change of a few hundredths of a pixel is within what a different seed can move", and the default gain is 0.1649 → 0.1446 (−0.020 px, documented and direct). The interpretation section does not connect the two, and the seed-variation claim is never measured; one of three unseen pairs got worse (0.0976 → 0.0987).
- **Consequence:** the learner is left to decide alone whether the fine-tune did anything, and the noise claim is asserted rather than shown.
- **Evidence:** documented execution (release-verification record) and direct execution (`default.baseline_epe`, `default.adapted_epe`, `default.new_data`).
- **Recommended correction:** add a checkpoint asking whether −0.02 px is distinguishable from seed variation, and either measure it (2–3 `SEED` values, cheap on T4) or soften the claim to "may be".
- **Acceptance check:** the interpretation cell states the gain relative to a measured or explicitly unmeasured variation, and the per-pair "adapted worse" case is named.
- **Spec:** framework dimension 3/5; GDL9, GDL14.

### RF-m4 — Minor: BYOD dataset results are not recorded and the reload is not checked

- **Cell/section:** cell 29, `USE_BYOD_DATASET` branch.
- **Observed issue:** the branch prints an unlabelled tuple `{'epe': (zero, pretrained, adapted), '3px': (…)}`, saves and loads the adapter but does not compare reloaded flow (the default path does), and writes nothing to `raft_optical_flow_result.json` or the metrics CSV. With the documented minimum of 2 records the held-out split is 1 pair and nothing warns about it. There is no new-data inference step for BYOD.
- **Evidence:** direct execution, local CPU: 4-record and 2-record BYOD runs completed (`{'epe': (7.0008, 0.5539, 0.3013), …}`), `result_json_mentions_byod: false`; only `byod_raft_adapter.safetensors` was added to `outputs/`.
- **Recommended correction:** reuse the default path's reload-equivalence check and table printer for BYOD, write a `byod_result.json`, warn when the held-out split has fewer than ~5 pairs.
- **Acceptance check:** a BYOD dataset run writes a JSON with zero-flow / pretrained / adapted metrics, the split sizes and a reload difference within tolerance; a 2-record run prints a small-split warning.
- **Spec:** DAT13, DAT14 (new-data inference/export), REL12.

### RF-m5 — Minor (spec): Declares NOTEBOOK_SPEC 2.1; current is 2.2

- **Observed issue:** `metadata.dimer.notebook_spec = '2.1'` and the intro/README cite 2.1. 2.2 supersedes the same-day 2.1 text and adds GDL1–GDL15 as SHOULDs.
- **Recommended correction:** regenerate against 2.2 after RF-M2.
- **Acceptance check:** metadata and the intro name 2.2, and `validate_release_assets.py` passes.
- **Spec:** versioning (§32).

### RF-m6 — Minor (release gate): REL12 not exercised

- **Observed issue:** the repository states REL12 is pending; no hosted record of the BYOD branch with representative input plus one incompatible `pairs.json`. This review's local CPU run of both cases succeeded/refused as designed, which is not hosted evidence.
- **Acceptance check:** a hosted run recorded in `docs/release-verification.md` per release step 7.
- **Spec:** REL12 (MUST for release).

### Suggestions

- **RF-S1:** The blank-pair probe stores `epe_vs_zero_truth` while §5 says the true motion of a blank pair is undetermined; rename it `mean_predicted_displacement_px` (`tools/notebook_template.py:195`).
- **RF-S2:** The update-count sweep shows 32 updates worse than 12 (0.1893 vs 0.1625) and the blank pair inventing up to 0.86 px; both are recorded in the release record but not discussed in the notebook. A "What to notice" note would turn them into teaching points.

## 4. Positive findings and non-findings

- `pipe` is never fine-tuned; `adapter` is a fresh `from_pretrained` copy, so the pretrained reference column and the unseen-pair comparison stay valid on a default run (direct execution).
- No train/held-out near-duplicates by construction: each pair is rendered from its own seed (`seed*100000+i`), unseen pairs use seeds 9,900,000+; the split asserts no id overlap.
- BYOD limits in the prose (2–2,000 records, sides 128–1024 px, ≤786,432 px) match the code (`split_pairs`, `MAX_RECORDS`, `validate_pair`); the 1-record case is refused with "at least 2 records are required to split".
- Non-multiple-of-8 BYOD frames (130×133) go through fine-tune, evaluate and export (padding handled).
- No unconditional sample assert blocks BYOD: the reload and `.flo` asserts concern the sample and run before the optional section.
- CPU numbers are not quoted as facts; the update-count default and limits come from the module constants.

## 5. Promise-to-evidence matrix

| Promise | Cell | Observable | Basis | Result |
|---|---|---|---|---|
| One-pass Run all with pinned install | 3 | no restart | documented | **Fails** (restart needed) — RF-M1 |
| Digest-verified checkpoint | 9 | `verified_files` 1 | documented + direct | Pass |
| Flow on rendered pair vs zero-flow | 11 | EPE 0.1625 vs 6.8937 | documented + direct | Pass |
| Update-count sweep | 13 | 1/4/12/32 rows | documented + direct | Pass (32 worse than 12, not discussed) |
| Static / blank probes | 13 | 0.0095 / 0.2425 px | documented + direct | Pass |
| Dataset validation + leak-free split | 15, 17 | 30/10, assertion | documented + direct | Pass |
| Bounded fine-tune, held-out re-score | 19, 21 | 0.1649 → 0.1446 | documented + direct | Pass; interpretation weak (RF-m3) |
| Unseen-pair inference | 23 | 3 rows | documented + direct | Pass |
| Export, fresh reload, equivalence | 25 | diff 0.0 ≤ 1e-3 px | documented + direct | Pass |
| 8 output files | 27 | listing | documented + direct | Pass |
| "Change a parameter and compare" | 30 | stacked rerun | direct | **Confounded** — RF-M3 |
| BYOD frame pair | 29 | `not-measurable` report | direct (CPU) | Pass locally; hosted not verified |
| BYOD dataset full workflow | 29 | export + reload | direct (CPU) | Runs; not recorded, no reload check — RF-m4 |

## 6. Readiness

**Needs revision.** Open Majors RF-M1 (RUN10/ENV6 MUST), RF-M2, RF-M3; REL12 MUST unmet (RF-m6). Gates in order: uv isolated environment (RF-M1) → guided layer with a reset-safe exercise (RF-M2, RF-M3) → minors → regenerate against 2.2 → hosted default run with no restart plus the REL12 BYOD exercise recorded in `docs/release-verification.md`.

## 7. Probe inventory

`raft_optical_flow_colab_Review_Probes.zip`: `run_probes.py` (static scan; default path cell-by-cell; Try-next rerun with and without reset; nine BYOD cases), `results.json`, `source_manifest.json` (sha256 of the notebook, generator, modules, docs and NOTEBOOK_SPEC 2.2). Probes ran on local CPU with `CUDA_VISIBLE_DEVICES=-1`, ~13 min total. The first full run's BYOD "incompatible `pairs.json`" case was invalidated by a probe bug (backslash in a substituted Windows path); the default and reuse sections were re-run after the fix and the active-learning section carried over (`phase_note` in `results.json`).

**Verified vs inferred.** Verified by direct execution: default-path completion and numbers on CPU, stacked-rerun behaviour, BYOD acceptance/refusals, the Colab-only upload failure. Verified from documented execution: the hosted restart. Inferred: that learners would rerun only cell 19 for the exercise (the field lives there, and no instruction says otherwise). **Most likely to be wrong:** RF-M3's Major rating — the numerical effect on this synthetic data is small (0.1048 vs 0.1089), and a reviewer could reasonably call it Minor.
