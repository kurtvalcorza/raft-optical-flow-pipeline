# Fleet-sweep fixes: `raft_optical_flow_colab.ipynb` (2026-10-05)

A targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report; each flag was first
confirmed in the cell source at `main` `9e8462e`. All changes are made in the generator (`tools/build_notebook.py`,
`tools/notebook_template.py`); the notebook is regenerated. Status and release labels are unchanged.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 pip-installed the pins into the kernel and raised "Restart the runtime" on stale modules (the recorded Kaggle run stopped there and was restarted). The generator is now the fleet's shared `build_notebook.py/2.2` with this repository's NOTEBOOK_SPEC 2.1 strings and its `weights_host` support (checkpoint on download.pytorch.org pinned by SHA-256) carried over; with the old generator's output compared cell by cell, only the generator version and the "pinned Python distributions" wording changed. The template opts in: one kernel cell verifies and runs the pinned `uv` 0.12.15, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (33 packages compiled from the unchanged pyproject pins, `--require-hashes --only-binary :all:`), keys the folder on the lock digest and reuses it, keeps a live worker on re-run, forces `MPLBACKEND=Agg` and drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1; generator, template, validator, new lock | `test_swp_r_*` (3 tests) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED with 1 of 9 guided markers. Added audience, Input → Model → Output, How to use, roadmap, Predict prompts (Sections 4, 5, 7–10), What to notice + Check your reasoning after Sections 4, 5, 7–11 quoting the recorded Kaggle T4 run of 2026-09-25 (demo EPE 0.1625 vs zero flow 6.8937; 32 updates worse at 0.1893; blank-pair spurious motion up to 0.8614 px; held-out EPE 0.1649 → 0.1446; one unseen pair 0.0976 → 0.0987), Troubleshooting, Glossary, Conclusion template; infrastructure labelled and collapsed. | opening, Sections 4–11 markdown, closing | `test_swp_g_*` (2 tests) |
| SWP-A (quality asserts) | Not flagged | The remaining asserts check split disjointness, reload parity and `.flo` round-trip — contract checks, kept. | — | — |
| SWP-F (frozen re-run) | Not present | The fine-tune already runs on its own fresh copy (`adapter`), so `pipe` stays the pretrained reference. | — | — |
| SWP-B (BYOD upload only) | Not present | Both BYOD branches already have location fields (`BYOD_IMAGE1_PATH`, `BYOD_IMAGE2_PATH`, `BYOD_DATASET_DIR`). | — | — |

## User-visible changes

- Section 1 installs nothing into the kernel and never asks for a restart (first build takes a few minutes; reused afterwards). Linux x86_64 only.
- Guided-layer cells; infrastructure collapsed.

## Verification (offline; not clean-runtime evidence)

- No model stage can run here (download.pytorch.org is blocked). The Section 1 cell runs for real against a stand-in environment (reuse, idempotent re-run, Agg backend). Plumbing evidence, not model evidence.
- CI installs CPU torch from `download.pytorch.org`, blocked here; torch was not installed. `pytest` with the other CI pins: 36 passed, 3 skipped before → 41 passed, 3 skipped after.
- `build_notebook.py --check` up to date; `validate_release_assets.py` PASS; `ruff check src tests tools` clean.
- Sweep re-check on the regenerated notebook: isolated runtime, guided markers 9/9, quality asserts 0.

## Remaining gates

- A hosted **Run all in one pass** in a fresh Colab T4 runtime (no restart expected), then a re-run of the Section 12 output cell.
- The REL12 BYOD run (still pending from before this fix).
- A full Notebook Review Framework v1 review has not been done.
