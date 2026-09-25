import builtins

import pytest

from raft_optical_flow_pipeline import pipeline as pipeline_module

TEST_REVISION = "0123456789abcdef" * 4


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in {"torch", "torchvision"}:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.fixture
def pinned(monkeypatch):
    """Pretend the package is pinned to TEST_REVISION, so snapshot checks can run on synthetic manifests."""
    monkeypatch.setattr(pipeline_module, "MODEL_REVISION", TEST_REVISION)
    return TEST_REVISION
