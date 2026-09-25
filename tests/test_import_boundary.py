"""Import-boundary contract: rejected requests never import model libraries.

Valid snapshots still reach them.
"""

import hashlib
import json

import pytest

from raft_optical_flow_pipeline import pipeline as pipeline_module
from raft_optical_flow_pipeline.pipeline import (
    MANIFEST_NAME,
    MODEL_ID,
    WEIGHTS_FILE,
    WEIGHTS_URL,
    RaftPipeline,
)

_PAYLOAD = b"not a real checkpoint"


def _snapshot(root, revision, tamper=False):
    (root / WEIGHTS_FILE).write_bytes(_PAYLOAD)
    digest = "0" * 64 if tamper else hashlib.sha256(_PAYLOAD).hexdigest()
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [{"path": WEIGHTS_FILE, "url": WEIGHTS_URL, "bytes": len(_PAYLOAD), "sha256": digest}],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_unpinned_package_refuses_before_model_imports(tmp_path, monkeypatch, forbid_model_imports):
    monkeypatch.setattr(pipeline_module, "MODEL_REVISION", "unpinned")
    with pytest.raises(RuntimeError, match="no pinned checkpoint digest"):
        RaftPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_from_pretrained_refuses_without_snapshot_before_model_imports(
    tmp_path, pinned, forbid_model_imports
):
    with pytest.raises(FileNotFoundError, match="no checkpoint manifest"):
        RaftPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_tampered_snapshot_before_model_imports(
    tmp_path, pinned, forbid_model_imports
):
    _snapshot(tmp_path, pinned, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        RaftPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_valid_snapshot_reaches_model_import(tmp_path, pinned, forbid_model_imports):
    _snapshot(tmp_path, pinned)
    with pytest.raises(AssertionError, match="model dependency imported before rejection"):
        RaftPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)
