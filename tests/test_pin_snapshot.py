"""tools/pin_snapshot.py against a fake download: no network, no weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "weights/raft-large-c-t-skht-v2/dimer-base-manifest.json"
MODULE = "src/raft_optical_flow_pipeline/pipeline.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pin_snapshot", ROOT / "tools" / "pin_snapshot.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pin_snapshot = _load_tool()


def _copy_repo(tmp_path: Path) -> Path:
    for relative in (
        "tools/notebook_template.py",
        "README.md",
        "MODEL_CARD.md",
        "STATUS.md",
        "docs/WEIGHTS.md",
    ):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copytree(ROOT / "weights", tmp_path / "weights", ignore=shutil.ignore_patterns("*.pth"))
    return tmp_path


def _setup(tmp_path: Path, data: bytes) -> Path:
    """Copy the repo and rename the manifest file so its prefix is the payload's real digest prefix."""
    root = _copy_repo(tmp_path)
    manifest = json.loads((root / MANIFEST).read_text())
    prefix = hashlib.sha256(data).hexdigest()[:8]
    name = f"raft_large_C_T_SKHT_V2-{prefix}.pth"
    manifest["files"][0].update(path=name, url=f"https://download.pytorch.org/models/{name}")
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return root


def _fake_download(data: bytes, calls: list[str]):
    def download(url, target):
        calls.append(url)
        target.write_bytes(data)

    return download


DATA = b"\x00" * 64


def test_pin_writes_digest_size_and_module_revision(tmp_path):
    root = _setup(tmp_path, DATA)
    calls: list[str] = []
    checked: list[Path] = []
    code = pin_snapshot.pin(
        root, download=_fake_download(DATA, calls), load_check=lambda p: checked.append(p) or "ok"
    )
    assert code == 0
    digest = hashlib.sha256(DATA).hexdigest()
    manifest = json.loads((root / MANIFEST).read_text())
    [entry] = manifest["files"]
    assert calls == [entry["url"]] and len(checked) == 1
    assert manifest["revision"] == entry["sha256"] == digest
    assert entry["bytes"] == manifest["totalBytes"] == len(DATA)
    assert (root / "weights/raft-large-c-t-skht-v2" / entry["path"]).read_bytes() == DATA
    assert f'MODEL_REVISION = "{digest}"' in (root / MODULE).read_text()


def test_pin_refuses_a_digest_that_disagrees_with_the_file_name(tmp_path):
    root = _setup(tmp_path, DATA)
    before = (root / MANIFEST).read_text()
    module_before = (root / MODULE).read_text()
    code = pin_snapshot.pin(root, download=_fake_download(b"tampered", []), load_check=lambda p: "ok")
    assert code == 1
    assert (root / MANIFEST).read_text() == before
    assert (root / MODULE).read_text() == module_before


def test_pin_refuses_a_checkpoint_that_does_not_load(tmp_path):
    root = _setup(tmp_path, DATA)
    before = (root / MANIFEST).read_text()
    module_before = (root / MODULE).read_text()

    def broken(_path):
        raise RuntimeError("Missing key(s) in state_dict")

    assert pin_snapshot.pin(root, download=_fake_download(DATA, []), load_check=broken) == 1
    assert (root / MANIFEST).read_text() == before
    assert (root / MODULE).read_text() == module_before


def test_dry_run_writes_nothing(tmp_path):
    root = _setup(tmp_path, DATA)
    before = (root / MANIFEST).read_text()
    module_before = (root / MODULE).read_text()
    assert (
        pin_snapshot.pin(root, dry_run=True, download=_fake_download(DATA, []), load_check=lambda p: "ok")
        == 0
    )
    assert (root / MANIFEST).read_text() == before
    assert (root / MODULE).read_text() == module_before


def test_committed_manifest_names_the_torchvision_prefix():
    entry = json.loads((ROOT / MANIFEST).read_text())["files"][0]
    assert pin_snapshot.PREFIX.search(entry["path"]).group(1) == "ff5fadd5"
    assert entry["url"].endswith("/" + entry["path"])


def test_load_check_refuses_a_state_dict_that_does_not_fit(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    path = tmp_path / "wrong.pth"
    torch.save({"feature_encoder.convnormrelu.0.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="state_dict"):
        pin_snapshot._load_check(path)
