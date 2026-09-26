"""End-to-end adaptation on a random-weight RAFT-Large at 128 px: no checkpoint, no network.

The model is torchvision's own ``raft_large`` architecture, the one the pinned checkpoint loads into; the
tests use small frames and few refinement steps so a CPU step takes well under a second. They exercise
the real estimation, evaluation, sequence-loss fine-tuning, adapter export and adapter reload code paths.
They say nothing about flow quality.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from safetensors.torch import save_file  # noqa: E402

from raft_optical_flow_pipeline import (  # noqa: E402
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_KEY,
    RaftPipeline,
    build_model,
    flow_dataset,
    sequence_loss,
    split_pairs,
)
from raft_optical_flow_pipeline import pipeline as pipeline_module  # noqa: E402

UPDATES = 3


def _pipeline(seed: int = 0) -> RaftPipeline:
    torch.manual_seed(seed)
    return RaftPipeline(model=build_model().eval(), device="cpu", source="random")


@pytest.fixture(scope="module")
def data():
    return split_pairs(flow_dataset(6, size=(128, 128)), train_fraction=0.67, seed=0)


def test_architecture_matches_torchvision_metadata():
    from torchvision.models.optical_flow import Raft_Large_Weights

    model = build_model()
    assert (
        sum(p.numel() for p in model.parameters())
        == Raft_Large_Weights.C_T_SKHT_V2.meta["num_params"]
        == 5_257_536
    )
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in model.context_encoder.modules())


def test_estimate_pads_to_a_multiple_of_eight_and_crops_back():
    from PIL import Image

    pipe = _pipeline()
    first, second = Image.new("RGB", (133, 130), "gray"), Image.new("RGB", (133, 130), "white")
    out = pipe.estimate(first, second, num_flow_updates=2)
    assert out["flow"].shape == (130, 133, 2) and out["num_flow_updates"] == 2


def test_sequence_loss_weights_later_iterations_more():
    truth = torch.zeros(1, 2, 8, 8)
    valid = torch.ones(1, 8, 8, dtype=torch.bool)
    early, late = torch.ones(1, 2, 8, 8), torch.zeros(1, 2, 8, 8)
    assert float(sequence_loss([early, late], truth, valid, gamma=0.5)) == pytest.approx(0.5)
    assert float(sequence_loss([late, early], truth, valid, gamma=0.5)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="gamma"):
        sequence_loss([early], truth, valid, gamma=1.0)


def test_finetune_holds_batchnorm_and_frozen_encoders_then_reloads(tmp_path, data):
    train, held = data
    pipe = _pipeline()
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    baseline = pipe.evaluate(held, num_flow_updates=UPDATES)
    assert baseline["zero_flow_baseline"]["epe"] > 0 and baseline["adapted"] is False
    run = pipe.finetune(train, epochs=1, batch_size=2, num_flow_updates=UPDATES, seed=1)
    after = pipe.model.state_dict()
    assert all(
        torch.equal(before[k], after[k]) for k in before if k.startswith(pipeline_module.ENCODER_PREFIXES)
    )
    assert any("running_mean" in k and k.startswith("context_encoder.") for k in before)
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("update_block."))
    assert (
        0 < run["trainable_parameters"] < run["total_parameters"] and run["batchnorm"] == "frozen (eval mode)"
    )

    path = tmp_path / "adapter.safetensors"
    descriptor = pipe.save_artifact(path, notes="small")
    assert descriptor["format"] == ARTIFACT_FORMAT and descriptor["frozen_prefixes"] == list(
        pipeline_module.ENCODER_PREFIXES
    )
    fresh = _pipeline()  # same seed: identical frozen encoders
    fresh.apply_artifact(path)
    record = held[0]
    first = pipe.estimate(record["image1"], record["image2"], num_flow_updates=UPDATES)["flow"]
    second = fresh.estimate(record["image1"], record["image2"], num_flow_updates=UPDATES)["flow"]
    assert abs(first - second).max() < 1e-4


def test_full_finetune_still_holds_batchnorm_statistics(data):
    train, _held = data
    pipe = _pipeline()
    stats = {k: v.clone() for k, v in pipe.model.state_dict().items() if "running_" in k}
    run = pipe.finetune(
        train, epochs=1, batch_size=2, num_flow_updates=UPDATES, seed=1, freeze_encoders=False
    )
    assert run["trainable_parameters"] == run["total_parameters"] and run["frozen_prefixes"] == []
    assert all(torch.equal(v, pipe.model.state_dict()[k]) for k, v in stats.items())


def test_without_the_hold_batchnorm_statistics_drift(monkeypatch, data):
    train, _held = data
    monkeypatch.setattr(pipeline_module, "_hold_batchnorm", lambda module: None)
    pipe = _pipeline()
    stats = {k: v.clone() for k, v in pipe.model.state_dict().items() if "running_mean" in k}
    pipe.finetune(train, epochs=1, batch_size=2, num_flow_updates=UPDATES, seed=1)
    assert any(not torch.equal(v, pipe.model.state_dict()[k]) for k, v in stats.items())


def test_apply_artifact_refuses_mismatched_adapters(tmp_path, data):
    pipe = _pipeline()
    tensor = {"update_block.flow_head.conv2.bias": torch.zeros(2)}
    forged = tmp_path / "forged.safetensors"
    save_file(tensor, str(forged), metadata={"format": "other", "model_id": MODEL_ID, "model_key": MODEL_KEY})
    with pytest.raises(ValueError, match="artifact format"):
        pipe.apply_artifact(forged)
    partial = tmp_path / "partial.safetensors"
    metadata = {
        "format": ARTIFACT_FORMAT,
        "model_id": MODEL_ID,
        "model_revision": pipeline_module.MODEL_REVISION,
        "model_key": MODEL_KEY,
        "frozen_prefixes": json.dumps(list(pipeline_module.ENCODER_PREFIXES)),
        "base_state_digest": "",
        "notes": "",
    }
    save_file(tensor, str(partial), metadata=metadata)
    with pytest.raises(ValueError, match="missing trainable tensors"):
        pipe.apply_artifact(partial)
    with pytest.raises(RuntimeError, match="not been fine-tuned"):
        pipe.save_artifact(tmp_path / "none.safetensors")
    with pytest.raises(ValueError, match="batch_size"):
        pipe.finetune(data[0], batch_size=0)
