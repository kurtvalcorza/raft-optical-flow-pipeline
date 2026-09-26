import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from raft_optical_flow_pipeline import (
    ARTIFACT_FORMAT,
    DEFAULT_WEIGHTS_DIR,
    INPUT_SCHEMA,
    MAX_FLOW,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    WEIGHTS_FILE,
    WEIGHTS_URL,
    RaftPipeline,
    aggregate_metrics,
    blank_pair,
    evaluation_report,
    flow_dataset,
    flow_metrics,
    is_pinned,
    moving_shapes_pair,
    read_flo,
    read_flow_records,
    split_pairs,
    stage_missing_files,
    static_pair,
    validate_dataset,
    validate_inputs,
    verify_snapshot,
    write_flo,
)
from raft_optical_flow_pipeline import pipeline as pipeline_module

HEX64 = re.compile(r"^[0-9a-f]{64}$")
REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "weights" / MODEL_KEY


def test_identity_constants_agree_with_the_committed_manifest():
    manifest = json.loads((SNAPSHOT / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert (
        MODEL_ID == "torchvision/raft_large" == manifest["modelId"]
        and manifest["format"] == "dimer_url_snapshot"
    )
    assert manifest["revision"] == pipeline_module.MODEL_REVISION
    assert is_pinned() == bool(HEX64.match(pipeline_module.MODEL_REVISION))
    assert DEFAULT_WEIGHTS_DIR == SNAPSHOT and ARTIFACT_FORMAT == "raft-adapter-v1"
    [entry] = manifest["files"]
    assert (entry["path"], entry["url"]) == (WEIGHTS_FILE, WEIGHTS_URL)
    assert WEIGHTS_FILE.endswith(f"-{pipeline_module.WEIGHTS_SHA256_PREFIX}.pth")
    if is_pinned():
        assert (
            entry["sha256"] == pipeline_module.MODEL_REVISION and manifest["totalBytes"] == entry["bytes"] > 0
        )
    else:
        assert entry["sha256"] is None and entry["bytes"] is None and manifest["totalBytes"] is None


def test_identity_matches_torchvision_weight_metadata():
    pytest.importorskip("torchvision")
    from torchvision.models.optical_flow import Raft_Large_Weights

    weights = Raft_Large_Weights.C_T_SKHT_V2
    assert weights == Raft_Large_Weights.DEFAULT and weights.url == WEIGHTS_URL
    assert str(weights) == pipeline_module.TORCHVISION_WEIGHTS
    assert tuple(weights.meta["min_size"]) == (MIN_IMAGE_SIDE, MIN_IMAGE_SIDE)


def test_unpinned_package_refuses_every_weight_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "MODEL_REVISION", "unpinned")
    for call in (
        lambda: verify_snapshot(tmp_path),
        lambda: stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None),
        lambda: RaftPipeline.from_pretrained(weights_dir=tmp_path),
    ):
        with pytest.raises(RuntimeError, match="pin_snapshot.py"):
            call()


def _write_snapshot(
    root: Path, revision: str, content: bytes, sha: str | None = None, size: int | None = None
) -> None:
    (root / WEIGHTS_FILE).write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [
            {
                "path": WEIGHTS_FILE,
                "url": WEIGHTS_URL,
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_and_stage_follow_the_manifest(tmp_path, pinned):
    _write_snapshot(tmp_path, pinned, b"checkpoint")
    assert verify_snapshot(tmp_path)["files"] == 1
    _write_snapshot(tmp_path, pinned, b"checkpoint", sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"checkpoint", size=3)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"payload")
    (tmp_path / WEIGHTS_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake(entry, root):
        fetched.append(entry["url"])
        (root / entry["path"]).write_bytes(b"payload")

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake) == [
        WEIGHTS_FILE
    ] and fetched == [WEIGHTS_URL]
    assert verify_snapshot(tmp_path)["files"] == 1
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["files"][0]["url"] = "https://example.invalid/evil.pth"
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    (tmp_path / WEIGHTS_FILE).unlink()
    with pytest.raises(ValueError, match="manifest URL"):
        stage_missing_files(tmp_path, allow_download=True, downloader=fake)


def test_synthetic_ground_truth_is_exact():
    for seed in range(6):
        record = moving_shapes_pair(seed)
        first = np.asarray(record["image1"]).astype(int)
        second = np.asarray(record["image2"]).astype(int)
        ys, xs = np.nonzero(record["valid"])
        u = record["flow"][ys, xs, 0].astype(int)
        v = record["flow"][ys, xs, 1].astype(int)
        assert np.array_equal(first[ys, xs], second[ys + v, xs + u])
        assert 0.8 < record["valid"].mean() <= 1.0
    assert moving_shapes_pair(3)["image1"].tobytes() == moving_shapes_pair(3)["image1"].tobytes()
    with pytest.raises(ValueError, match="multiples of 8"):
        moving_shapes_pair(0, size=(130, 128))


def test_flow_metrics_known_values():
    truth = np.zeros((8, 8, 2), np.float32)
    truth[..., 0] = 3.0
    assert flow_metrics(truth, truth)["epe"] == 0.0 and flow_metrics(truth, truth)[
        "angular_error_deg"
    ] == pytest.approx(0.0, abs=1e-6)
    zero = flow_metrics(np.zeros_like(truth), truth)
    assert zero["epe"] == pytest.approx(3.0) and zero["1px"] == 0.0 and zero["5px"] == 1.0
    assert zero["angular_error_deg"] == pytest.approx(np.degrees(np.arccos(1 / np.sqrt(10))), rel=1e-6)
    valid = np.zeros((8, 8), bool)
    valid[:2] = True
    assert flow_metrics(np.zeros_like(truth), truth, valid)["valid_pixels"] == 16
    huge = truth.copy()
    huge[0, 0] = (MAX_FLOW, 0)
    assert flow_metrics(huge, huge)["valid_pixels"] == 63
    with pytest.raises(ValueError, match="no valid pixel"):
        flow_metrics(truth, truth, np.zeros((8, 8), bool))
    with pytest.raises(ValueError, match="shape"):
        flow_metrics(truth[:4], truth)
    agg = aggregate_metrics(
        [{**zero, "valid_pixels": 10}, {**flow_metrics(truth, truth), "valid_pixels": 30}]
    )
    assert agg["epe"] == pytest.approx(1.5) and agg["pixel_weighted_epe"] == pytest.approx(0.75)


def test_validate_inputs_and_rejections():
    pair = moving_shapes_pair(0)
    manifest = validate_inputs(pair["image1"], pair["image2"], num_flow_updates=6)
    assert (
        manifest["schema"] == INPUT_SCHEMA
        and manifest["verdict"] == "accepted"
        and manifest["padding"] == [0, 0]
    )
    assert validate_inputs(Image.new("RGB", (130, 131)), Image.new("RGB", (130, 131)))["padding"] == [6, 5]
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    for first, second, kwargs, error, message in (
        ("x", pair["image2"], {}, TypeError, "PIL.Image.Image"),
        (pair["image1"], Image.new("RGB", (200, 200)), {}, ValueError, "differ in size"),
        (Image.new("RGB", (64, 200)), Image.new("RGB", (64, 200)), {}, ValueError, "MIN_IMAGE_SIDE"),
        (
            Image.new("RGB", (MAX_IMAGE_SIDE + 8, 200)),
            Image.new("RGB", (MAX_IMAGE_SIDE + 8, 200)),
            {},
            ValueError,
            "MAX_IMAGE_SIDE",
        ),
        (Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024)), {}, ValueError, "MAX_PIXELS"),
        (pair["image1"], pair["image2"], {"num_flow_updates": 0}, ValueError, "num_flow_updates"),
    ):
        with pytest.raises(error, match=message):
            validate_inputs(first, second, **kwargs)


def test_validate_dataset_reports_and_rejects():
    records = flow_dataset(3, size=(128, 128))
    manifest = validate_dataset(records, epochs=2)
    assert (
        manifest["verdict"] == "accepted"
        and manifest["n_records"] == 3
        and manifest["frame_sizes"] == [[128, 128]]
    )
    assert 0 < manifest["mean_displacement_px"] < 20 and manifest["findings"] == []
    big = dict(records[0], flow=np.full((128, 128, 2), MAX_FLOW, np.float32))
    assert any("ignored by the loss" in f for f in validate_dataset([big])["findings"])
    good = records[0]
    for record, message in (
        ({"image1": good["image1"], "image2": good["image2"]}, "must be a mapping"),
        (dict(good, flow=np.zeros((64, 64, 2))), "flow shape"),
        (dict(good, flow=np.full((128, 128, 2), np.nan)), "non-finite"),
        (dict(good, valid=np.zeros((128, 128), bool)), "selects no pixel"),
        (dict(good, valid=np.full((128, 128), 2)), "boolean"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset([record])
    with pytest.raises(ValueError, match="epochs"):
        validate_dataset(records, epochs=0)


def test_split_pairs_is_deterministic_and_disjoint():
    records = flow_dataset(8, size=(128, 128))
    train, held = split_pairs(records, train_fraction=0.75, seed=3)
    assert [r["id"] for r in train] == [r["id"] for r in split_pairs(records, train_fraction=0.75, seed=3)[0]]
    assert len(train) == 6 and len(held) == 2 and not {r["id"] for r in train} & {r["id"] for r in held}


def test_flo_round_trip_and_bad_files(tmp_path):
    flow = moving_shapes_pair(2)["flow"]
    write_flo(tmp_path / "a.flo", flow)
    assert np.array_equal(read_flo(tmp_path / "a.flo"), flow)
    (tmp_path / "bad.flo").write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError, match="magic"):
        read_flo(tmp_path / "bad.flo")
    with pytest.raises(ValueError, match="shape"):
        write_flo(tmp_path / "c.flo", np.zeros((4, 4)))


def _write_byod(root: Path, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pair = moving_shapes_pair(4, size=(128, 128))
    pair["image1"].save(root / "a.png")
    pair["image2"].save(root / "b.png")
    write_flo(root / "a.flo", pair["flow"])
    np.save(root / "a.npy", pair["flow"])
    Image.fromarray((pair["valid"] * 255).astype(np.uint8)).save(root / "a_valid.png")
    (root / "pairs.json").write_text(json.dumps(entries), encoding="utf-8")


def test_read_flow_records_reads_flo_npy_and_valid(tmp_path):
    _write_byod(
        tmp_path,
        [
            {"frame1": "a.png", "frame2": "b.png", "flow": "a.flo", "valid": "a_valid.png"},
            {"frame1": "a.png", "frame2": "b.png", "flow": "a.npy"},
        ],
    )
    records = read_flow_records(tmp_path)
    assert records[0]["flow"].shape == (128, 128, 2) and records[0]["valid"].dtype == bool
    assert np.array_equal(records[0]["flow"], records[1]["flow"]) and "valid" not in records[1]
    assert validate_dataset(records)["verdict"] == "accepted"


@pytest.mark.parametrize("name", ["../a.png", "/etc/passwd"])
def test_read_flow_records_refuses_paths_outside_the_directory(tmp_path, name):
    _write_byod(tmp_path / "set", [{"frame1": name, "frame2": "b.png", "flow": "a.flo"}])
    with pytest.raises(ValueError, match="relative|outside"):
        read_flow_records(tmp_path / "set")


def test_evaluation_report_verdicts():
    pair = moving_shapes_pair(5)
    report = evaluation_report({"flow": pair["flow"]}, pair["flow"], pair["valid"])
    assert report["verdict"] == "sample-sanity" and report["metrics"]["epe"] == 0.0
    [baseline] = report["baselines"]
    assert baseline["id"] == "zero-flow" and baseline["epe"] > 0
    missing = evaluation_report({"flow": pair["flow"]})
    assert missing["verdict"] == "not-measurable" and "zero-flow baseline" in missing["needs"]


def test_probe_pairs_have_zero_true_flow():
    for record in (static_pair(), blank_pair()):
        assert not record["flow"].any() and record["valid"].all()
        assert record["image1"].tobytes() == record["image2"].tobytes()
