"""Regression tests for the Notebook Review Framework v1 findings on raft_optical_flow_colab.ipynb (RF-M2, RF-M3,
RF-m1..RF-m5; review docs/reviews/2026-10-02-notebook-review/raft_optical_flow_colab_Review.md).

Every test needs only CI's dependencies (NumPy-level, no model, no checkpoint): the notebook's own cell sources are
executed with stand-ins where the model would be needed. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "raft_optical_flow_colab.ipynb"
METRIC_KEYS = ("epe", "1px", "3px", "5px", "angular_error_deg")


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _code(notebook: dict, marker: str) -> str:
    found = [_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and marker in _source(c)]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown_cells(notebook: dict) -> list[str]:
    return [_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown"]


def _index(cells: list[str], marker: str) -> int:
    hits = [i for i, text in enumerate(cells) if marker in text]
    assert len(hits) == 1, f"expected one markdown cell containing {marker!r}, found {len(hits)}"
    return hits[0]


def _metrics(value: float) -> dict:
    row = {key: value for key in METRIC_KEYS}
    row["pixel_weighted_epe"] = value
    return row


class _StandInPipeline:
    """Records how each fine-tune starts; no model."""

    built: list[_StandInPipeline] = []

    def __init__(self) -> None:
        self.adapted = False
        self.finetune_calls = 0

    @classmethod
    def from_pretrained(cls, weights_dir=None):
        pipe = cls()
        cls.built.append(pipe)
        return pipe

    @classmethod
    def load_artifact(cls, path, weights_dir=None):
        pipe = cls()
        pipe.adapted = True
        return pipe

    def finetune(self, records, *, epochs, batch_size, learning_rate, seed, freeze_encoders, progress=None):
        started_adapted = self.adapted
        self.finetune_calls += 1
        self.adapted = True
        losses = [0.4, 0.3, 0.2][:epochs]
        for i, loss in enumerate(losses):
            if progress:
                progress({"epoch": i + 1, "epochs": epochs, "loss": loss})
        return {"started_from_adapted": started_adapted, "freeze_encoders": freeze_encoders, "batchnorm": "frozen (eval mode)",
                "trainable_parameters": 3 if freeze_encoders else 5, "total_parameters": 5, "epochs": epochs, "batch_size": batch_size,
                "learning_rate": learning_rate, "optimizer": "AdamW", "loss": "sequence", "precision": "float32", "device": "cpu",
                "epoch_losses": losses}

    def evaluate(self, records, *, num_flow_updates):
        value = 0.1 if self.adapted else 0.2
        return {**_metrics(value), "zero_flow_baseline": _metrics(5.0), "adapted": self.adapted}

    def estimate(self, image1, image2, *, num_flow_updates):
        return {"flow": np.zeros((8, 8, 2), np.float32)}

    def save_artifact(self, path, *, notes=None):
        Path(path).write_bytes(b"stand-in adapter")
        return {"path": str(path), "sha256": "ab" * 32, "tensors": 1}


# --- RF-M3: re-running the fine-tune cell starts again from the verified checkpoint --------------------------------


def test_rf_m3_rerunning_section_8_repeats_one_finetune_from_the_pretrained_weights(notebook, capsys):
    source = _code(notebook, "run = adapter.finetune(")
    _StandInPipeline.built = []
    stale = _StandInPipeline()
    stale.adapted = True  # an adapter left adapted by an earlier run of the cell
    ns = {"RaftPipeline": _StandInPipeline, "WEIGHTS_DIR": "w", "train_records": [], "EPOCHS": 3, "SEED": 0, "adapter": stale, "json": json}
    exec(compile(source, "<section 8>", "exec"), ns)
    first = ns["adapter"]
    assert first is not stale and ns["run"]["started_from_adapted"] is False
    exec(compile(source.replace("FREEZE_ENCODERS = True", "FREEZE_ENCODERS = False"), "<section 8 rerun>", "exec"), ns)
    second = ns["adapter"]
    assert second is not first and second.finetune_calls == 1
    assert ns["run"]["started_from_adapted"] is False and ns["run"]["freeze_encoders"] is False
    assert len(ns["run"]["epoch_losses"]) == ns["EPOCHS"]
    out = capsys.readouterr().out
    assert out.count("rebuilt from the verified checkpoint") == 2 and '"epoch_losses"' in out


# --- RF-M2 / RF-M3: one Predict -> Change -> Run -> Observe -> Explain activity, with rerun instructions --------------


def test_rf_m2_activity_follows_the_five_steps_after_section_9(notebook):
    cells = _markdown_cells(notebook)
    activity = cells[_index(cells, "### Activity: does unfreezing the encoders help?")]
    steps = ["**Predict.**", "**Change.**", "**Run.**", "**Observe.**", "**Explain.**"]
    positions = [activity.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "Run the Section 8 cell, then the Section 9 cell" in activity and "<details><summary>Check your reasoning</summary>" in activity
    assert "rerun Sections 8–12" in activity  # restore the default run before exporting
    assert _index(cells, "## 9. Evaluate on the held-out split") < _index(cells, "### Activity") < _index(cells, "## 10. Inference on unseen pairs")
    closing = cells[_index(cells, "**Try next.**")]
    assert "Change `FREEZE_ENCODERS` to `False` and compare" not in closing


# --- RF-m1: the upload fallback outside Colab names the location fields --------------------------------------------


def _byod_namespace(tmp_path: Path) -> dict:
    def refuse(*args, **kwargs):
        raise ValueError("stand-in refusal")

    return {"static_pair": lambda: {"image1": None, "image2": None}, "validate_inputs": refuse, "validate_dataset": lambda *a, **k: {"records": 2},
            "np": np, "Path": Path, "OUTPUTS": tmp_path, "json": json}


@pytest.mark.parametrize(("flag", "fields"), [("USE_BYOD_IMAGE", "BYOD_IMAGE1_PATH and BYOD_IMAGE2_PATH"), ("USE_BYOD_DATASET", "BYOD_DATASET_DIR")])
def test_rf_m1_upload_outside_colab_names_the_location_fields(notebook, tmp_path, monkeypatch, flag, fields):
    source = _code(notebook, "def _upload_into(").replace(f"{flag} = False", f"{flag} = True")
    monkeypatch.setitem(sys.modules, "google", None)  # no google.colab: outside Colab
    with pytest.raises(ValueError) as caught:
        exec(compile(source, "<section 13>", "exec"), _byod_namespace(tmp_path))
    assert fields in str(caught.value) and "Google Colab" in str(caught.value)


# --- RF-m4: the BYOD dataset branch writes its result, checks the reload, warns on a tiny split ---------------------


@pytest.mark.parametrize(("n_held", "warned"), [(1, True), (5, False)])
def test_rf_m4_byod_dataset_branch_records_result_and_reload(notebook, tmp_path, capsys, n_held, warned):
    source = _code(notebook, "def _upload_into(").replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True").replace('BYOD_DATASET_DIR = ""', 'BYOD_DATASET_DIR = "byod-dir"')
    record = {"id": "pair-0", "image1": None, "image2": None, "flow": np.zeros((8, 8, 2), np.float32)}
    ns = _byod_namespace(tmp_path)
    ns.update({
        "read_flow_records": lambda d: [record] * (n_held + 1), "split_pairs": lambda r, train_fraction, seed: (r[:1], r[1:]),
        "RaftPipeline": _StandInPipeline, "WEIGHTS_DIR": "w", "NUM_FLOW_UPDATES": 12, "EPOCHS": 3, "BATCH_SIZE": 2, "LEARNING_RATE": 2e-5,
        "SEED": 0, "HOLDOUT": 0.25, "FREEZE_ENCODERS": True, "METRIC_KEYS": METRIC_KEYS, "TOLERANCE_PX": 1e-3,
        "NOTEBOOK_SOURCE": {"repository_revision": "x"}, "MODEL_ID": "torchvision/raft_large", "MODEL_REVISION": "f" * 64,
    })
    exec(compile(source, "<section 13>", "exec"), ns)
    out = capsys.readouterr().out
    assert ("WARNING: only 1 held-out pair(s)" in out) is warned
    assert "pixel_weighted_epe" in out and "pretrained" in out  # labelled table, not an unlabelled tuple
    result = json.loads((tmp_path / "byod_result.json").read_text(encoding="utf-8"))
    assert result["split"] == {"train": 1, "held_out": n_held, "seed": 0, "holdout": 0.25, "small_split_warning": warned}
    assert result["baseline"]["zero_flow_baseline"]["epe"] == 5.0 and result["baseline"]["epe"] == 0.2 and result["adapted"]["epe"] == 0.1
    assert result["reload_check"]["equivalent"] is True and result["reload_check"]["mean_endpoint_difference_px"] <= 1e-3


# --- RF-m2: pair counts are named as defaults; the runtime line cites recorded runs ---------------------------------


def test_rf_m2_pair_counts_are_stated_as_defaults(notebook):
    pattern = re.compile(r"\b(10|30|40)[ -](rendered |synthetic |held-out |training )?pairs?\b")
    offenders = []
    for text in _markdown_cells(notebook):
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
            if pattern.search(sentence) and not re.search(r"default|recorded run|N_PAIRS", sentence):
                offenders.append(sentence.strip()[:160])
    assert not offenders, offenders
    md = "\n".join(_markdown_cells(notebook))
    assert "Runtimes are not measured" not in md
    assert "recorded Kaggle Tesla T4 run of 25 September 2026" in md and "about 35 s" in md


# --- RF-m3: the gain is stated against an unmeasured variation, and the unseen pair that got worse is named ---------


def test_rf_m3_gain_is_read_against_unmeasured_seed_variation(notebook):
    cells = _markdown_cells(notebook)
    section_9 = cells[_index(cells, "## 9. Evaluate on the held-out split")]
    assert "is within what a different seed can move" not in section_9 and "does not measure" in section_9
    checkpoint = cells[_index(cells, "**Checkpoint: is a gain of about 0.02 px real?**")]
    assert "unmeasured" in checkpoint and "0.0976 → 0.0987" in checkpoint and "`SEED`" in checkpoint
    closing = cells[_index(cells, "**How large the gain is.**")]
    assert "0.0976 → 0.0987" in closing and "Seed variation was not measured" in closing


# --- RF-m5: declares NOTEBOOK_SPEC 2.2 -------------------------------------------------------------------------------


def test_rf_m5_declares_notebook_spec_2_2(notebook):
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
    md = "\n".join(_markdown_cells(notebook))
    assert "NOTEBOOK_SPEC 2.1" not in md and "Specification 2.1" not in md
