"""The model is served from the pinned safetensors conversion; the pickle source is read only to convert it.

The conversion tests swap the 5M-parameter RAFT architecture for a two-layer stand-in, so they run offline in
CI. The last test runs only where the real checkpoint and its conversion are staged under weights/.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from raft_optical_flow_pipeline import pipeline as pipeline_module
from raft_optical_flow_pipeline.pipeline import (
    CONVERTED_SHA256,
    CONVERTED_WEIGHTS_FILE,
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    MODEL_ID,
    WEIGHTS_FILE,
    WEIGHTS_URL,
    RaftPipeline,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_converted_file_is_pinned_safetensors():
    assert CONVERTED_WEIGHTS_FILE.endswith(".safetensors")
    assert len(CONVERTED_SHA256) == 64 and all(c in "0123456789abcdef" for c in CONVERTED_SHA256)
    assert CONVERTED_WEIGHTS_FILE.removesuffix(".safetensors") == WEIGHTS_FILE.removesuffix(".pth")


def test_validator_declares_the_converted_digest():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / "validate_release_assets.py"
    spec = importlib.util.spec_from_file_location("validate_release_assets", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    assert set(validator.EXTERNAL_WEIGHT_DIGESTS) == {CONVERTED_SHA256}


def test_serving_path_never_unpickles():
    source = inspect.getsource(RaftPipeline.from_pretrained)
    assert "torch.load" not in source and "load_file" in source and "strict=True" in source


def test_altered_conversion_is_refused_before_model_imports(tmp_path, pinned, forbid_model_imports):
    (tmp_path / CONVERTED_WEIGHTS_FILE).write_bytes(b"not the pinned conversion")
    with pytest.raises(ValueError, match="sha256"):
        RaftPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


@pytest.fixture
def stand_in(tmp_path, pinned, monkeypatch):
    """A pinned two-layer 'checkpoint' plus the digest its conversion must have."""
    torch = pytest.importorskip("torch")

    def build(num_classes=None):
        return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))

    torch.manual_seed(0)
    state = build().state_dict()
    root = tmp_path / "weights"
    root.mkdir()
    torch.save(state, root / WEIGHTS_FILE)
    entry = {"path": WEIGHTS_FILE, "url": WEIGHTS_URL, "bytes": (root / WEIGHTS_FILE).stat().st_size}
    entry["sha256"] = _sha(root / WEIGHTS_FILE)
    manifest = {"modelId": MODEL_ID, "revision": pinned, "files": [entry]}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    reference = tmp_path / "reference.safetensors"
    pipeline_module._write_converted(state, reference)
    monkeypatch.setattr(pipeline_module, "CONVERTED_SHA256", _sha(reference))
    monkeypatch.setattr(pipeline_module, "build_model", build)
    return torch, root, state


def test_conversion_is_byte_reproducible_and_strict(stand_in, tmp_path):
    _torch, root, _state = stand_in
    first = pipeline_module.convert_checkpoint(root)["converted"]["sha256"]
    (root / CONVERTED_WEIGHTS_FILE).unlink()
    second = pipeline_module.convert_checkpoint(root)["converted"]["sha256"]
    assert first == second == pipeline_module.CONVERTED_SHA256
    assert not list(root.glob("*.partial"))


def test_conversion_with_another_digest_is_refused_and_leaves_nothing(stand_in, monkeypatch):
    _torch, root, _state = stand_in
    monkeypatch.setattr(pipeline_module, "CONVERTED_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        pipeline_module.convert_checkpoint(root)
    assert not (root / CONVERTED_WEIGHTS_FILE).exists() and not list(root.glob("*.partial"))


def test_from_pretrained_converts_once_then_serves_without_the_source(stand_in, monkeypatch):
    torch, root, state = stand_in
    RaftPipeline.from_pretrained(device="cpu", weights_dir=root)  # converts
    assert (root / CONVERTED_WEIGHTS_FILE).is_file()
    (root / WEIGHTS_FILE).unlink()

    def refuse(*_args, **_kwargs):
        raise AssertionError("torch.load called on the serving path")

    monkeypatch.setattr(torch, "load", refuse)
    served = RaftPipeline.from_pretrained(device="cpu", weights_dir=root).model.state_dict()
    assert set(served) == set(state) and all(torch.equal(served[k], state[k]) for k in state)


@pytest.mark.skipif(
    not all((DEFAULT_WEIGHTS_DIR / name).is_file() for name in (WEIGHTS_FILE, CONVERTED_WEIGHTS_FILE)),
    reason="real checkpoint and conversion not staged",
)
def test_real_conversion_equals_the_source_checkpoint():
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    pipeline_module.verify_snapshot(DEFAULT_WEIGHTS_DIR)
    pipeline_module.verify_converted(DEFAULT_WEIGHTS_DIR)
    source = torch.load(DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    converted = load_file(str(DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_FILE))
    assert set(source) == set(converted)
    assert all(torch.equal(source[k], converted[k]) and source[k].dtype == converted[k].dtype for k in source)
