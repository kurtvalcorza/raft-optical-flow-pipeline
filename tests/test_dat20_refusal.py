"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both from the generated notebook:
- the router (install cell, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as
  one line, while every other exception keeps its traceback;
- the BYOD cell turns every user-input problem into `BYODRefusal` and removes the earlier run's
  results before it stops.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "raft_optical_flow_colab.ipynb"
BYOD_CELL_ID = "raft_optical_flow-40"


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _router_cell(cells: list[dict]) -> str:
    return next(_source(c) for c in cells if '_WORKER_SOURCE = r"""' in _source(c))


def _router_namespace(cells: list[dict], start: str) -> dict:
    source = _router_cell(cells)
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(cells, messages):
    namespace = _router_namespace(cells, "class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def test_router_relays_a_refusal_as_one_line_without_a_traceback(cells, capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime(cells, [("refusal", "etc: not accepted")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: etc: not accepted"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(cells, capsys):
    namespace, runtime = _kernel_runtime(cells, [("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(cells, capsys):
    _, runtime = _kernel_runtime(cells, [("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")
    assert runtime._send.sent == [("run", "print('hello')\n")]


def test_router_has_the_refusal_channel_before_the_error_branch(cells):
    source = _router_cell(cells)
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    assert "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/." in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


# --- router, end to end with a real worker (POSIX: the router uses pass_fds and start_new_session) ----------------


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(cells, capsys):
    namespace = _router_namespace(cells, '_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('etc: not accepted')\n")
        assert caught.value._render_traceback_() == ["Refused: etc: not accepted"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


# --- BYOD cell ----------------------------------------------------------------------------------------------------

CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _byod_cell(cells: list[dict]) -> dict:
    (cell,) = [c for c in cells if c.get("id") == BYOD_CELL_ID]
    return cell


def test_byod_refusal_is_carried_verbatim_and_the_cell_id_is_stable(cells):
    assert CARRIED in _source(_byod_cell(cells))


def _one_line(exc: BaseException, file_name: str, rule: str) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert line.startswith(f"Refused: {file_name}: "), line
    assert rule in line and "\n" not in line and "Traceback" not in line, line
    assert "again" in line, line
    return line


def _colab(monkeypatch, uploaded):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


# --- raft: Section 13, both branches ------------------------------------------------------------------------------

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import raft_optical_flow_pipeline as raft  # noqa: E402

EARLIER_IMAGE = {"byod_frames": "old", "byod_result": "old", "byod_report": "old"}
EARLIER_DATASET = {"byod_records": "old", "byod_manifest": "old", "byod_adapted": "old", "byod_reload_check": "old", "byod_export": "old"}


def _cell_code(cells, *, image=("", ""), dataset=None) -> str:
    source = _source(_byod_cell(cells))
    if image != ("", ""):
        source = source.replace("USE_BYOD_IMAGE = False", "USE_BYOD_IMAGE = True")
        source = source.replace('BYOD_IMAGE1_PATH = ""', f"BYOD_IMAGE1_PATH = {image[0]!r}").replace('BYOD_IMAGE2_PATH = ""', f"BYOD_IMAGE2_PATH = {image[1]!r}")
    if dataset is not None:
        source = source.replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True").replace('BYOD_DATASET_DIR = ""', f"BYOD_DATASET_DIR = {dataset!r}")
    return source


def _namespace(tmp_path: Path) -> dict:
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    for name in ("byod_flow.flo", "byod_flow.png", "byod_result.json", "byod_raft_adapter.safetensors"):
        (outputs / name).write_bytes(b"earlier run")
    ns = {name: getattr(raft, name) for name in raft.__all__}
    ns.update({"np": np, "Image": Image, "Path": Path, "json": json, "OUTPUTS": outputs, "EPOCHS": 3, "HOLDOUT": 0.25, "SEED": 0, "NUM_FLOW_UPDATES": 12, "pipe": "kept"})
    ns.update(EARLIER_IMAGE)
    ns.update(EARLIER_DATASET)
    return ns


def _run_refused(code: str, ns: dict) -> BaseException:
    with pytest.raises(ValueError) as caught:
        exec(compile(code, "<section 13>", "exec"), ns)
    return caught.value


def _etc(tmp_path: Path) -> Path:
    etc = tmp_path / "etc"
    etc.mkdir()
    (etc / "hostname").write_text("runtime\n", encoding="utf-8")
    return etc


def test_scripted_refusal_dataset_dir_without_pairs_json(cells, tmp_path, monkeypatch):
    """The hosted refusal run sets USE_BYOD_DATASET=True;BYOD_DATASET_DIR='/etc': a folder with no pairs.json."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _namespace(tmp_path)
    exc = _run_refused(_cell_code(cells, dataset=str(_etc(tmp_path))), ns)
    _one_line(exc, "etc", "pairs.json not found")
    assert not set(EARLIER_DATASET) & set(ns), "an earlier run's dataset results survived the refusal"
    assert not (ns["OUTPUTS"] / "byod_result.json").exists() and not (ns["OUTPUTS"] / "byod_raft_adapter.safetensors").exists()
    assert set(EARLIER_IMAGE) <= set(ns) and (ns["OUTPUTS"] / "byod_flow.png").exists()  # the other branch is untouched
    assert ns["pipe"] == "kept"


@pytest.mark.skipif(not Path("/etc").is_dir() or Path("/etc/pairs.json").exists(), reason="needs a POSIX /etc without pairs.json, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_directory(cells, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    exc = _run_refused(_cell_code(cells, dataset="/etc"), _namespace(tmp_path))
    _one_line(exc, "etc", "/etc/pairs.json not found")


@pytest.mark.parametrize(
    ("make", "file_name", "rule"),
    [
        (lambda d: (d / "pairs.json").write_text("{}", encoding="utf-8"), "ds", "pairs.json must hold a non-empty list"),
        (lambda d: (d / "pairs.json").write_text("not json", encoding="utf-8"), "ds", "Expecting value"),
        (lambda d: (d / "pairs.json").write_text('[{"frame1": "../x.png", "frame2": "b.png", "flow": "a.flo"}]', encoding="utf-8"), "ds", "must be relative"),
        (lambda d: (d / "pairs.json").write_text('[{"frame1": "a.png", "frame2": "b.png", "flow": "a.flo"}]', encoding="utf-8"), "ds", "a.png"),
    ],
    ids=["not a list", "not json", "escapes the folder", "missing frame"],
)
def test_dataset_problems_are_one_refused_line(cells, tmp_path, monkeypatch, make, file_name, rule):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    folder = tmp_path / "ds"
    folder.mkdir()
    make(folder)
    ns = _namespace(tmp_path)
    _one_line(_run_refused(_cell_code(cells, dataset=str(folder)), ns), file_name, rule)
    assert not set(EARLIER_DATASET) & set(ns)


def test_cancelled_dataset_upload_is_one_refused_line(cells, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _colab(monkeypatch, {})
    ns = _namespace(tmp_path)
    _one_line(_run_refused(_cell_code(cells, dataset=""), ns), "(no file)", "cancelled or empty")
    assert not set(EARLIER_DATASET) & set(ns)


def test_dataset_upload_outside_colab_is_one_refused_line(cells, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    _one_line(_run_refused(_cell_code(cells, dataset=""), _namespace(tmp_path)), "(no file)", "BYOD_DATASET_DIR")


@pytest.mark.parametrize(
    ("frames", "file_name", "rule"),
    [
        (("missing.png", "b.png"), "missing.png", "is not a file"),
        (("notes.png", "b.png"), "notes.png + b.png", "cannot identify image file"),
        (("small.png", "small.png"), "small.png + small.png", "MIN_IMAGE_SIDE"),
    ],
    ids=["missing frame", "not an image", "too small"],
)
def test_frame_pair_problems_are_one_refused_line(cells, tmp_path, monkeypatch, frames, file_name, rule):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "notes.png").write_text("not an image", encoding="utf-8")
    Image.new("RGB", (64, 64)).save(tmp_path / "small.png")
    Image.new("RGB", (256, 256)).save(tmp_path / "b.png")
    ns = _namespace(tmp_path)
    paths = tuple(str(tmp_path / f) for f in frames)
    _one_line(_run_refused(_cell_code(cells, image=paths), ns), file_name, rule)
    assert not set(EARLIER_IMAGE) & set(ns), "an earlier run's frame-pair results survived the refusal"
    assert not (ns["OUTPUTS"] / "byod_flow.flo").exists() and not (ns["OUTPUTS"] / "byod_flow.png").exists()
    assert set(EARLIER_DATASET) <= set(ns)


def test_cancelled_frame_upload_is_one_refused_line(cells, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _colab(monkeypatch, None)
    _one_line(_run_refused(_cell_code(cells, image=("", "x")), _namespace(tmp_path)), "(no file)", "cancelled or empty")


def test_one_uploaded_frame_is_one_refused_line(cells, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _colab(monkeypatch, {"only.png": b"x"})
    _one_line(_run_refused(_cell_code(cells, image=("", "x")), _namespace(tmp_path)), "only.png", "exactly two frames, got 1")


def test_genuine_internal_error_keeps_its_traceback(cells, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _namespace(tmp_path)

    def broken(_directory):
        raise RuntimeError("internal bug")

    ns["read_flow_records"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        exec(compile(_cell_code(cells, dataset=str(tmp_path)), "<section 13>", "exec"), ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert ns["byod_adapted"] == "old"  # nothing is cleared for an internal error


def test_default_path_does_not_refuse(cells, tmp_path, capsys):
    ns = _namespace(tmp_path)
    exec(compile(_cell_code(cells), "<section 13>", "exec"), ns)
    out = capsys.readouterr().out
    assert "BYOD frame-pair branch is off" in out and "BYOD dataset branch is off" in out
    assert "Refused:" not in out and ns["byod_result"] == "old"
