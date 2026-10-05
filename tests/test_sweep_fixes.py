"""Regression tests for the 2026-10-05 fleet-sweep fixes of raft_optical_flow_colab.ipynb (SWP-R, SWP-G).

Every test needs only CI's dependencies (NumPy-level, no model, no torch): the notebook's own cell sources are executed
with stand-ins where a model would be needed. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "raft_optical_flow_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(notebook: dict, marker: str) -> str:
    found = [_source(c) for c in _code_cells(notebook) if marker in _source(c)]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart guard, environment reuse, idempotent Section 1 ----------------------


def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(_source(c) for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in _source(c)]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = _source(kernel[0])
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", "_isolated_environment_ready()", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source, needed


def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock([p for p in build._pins(ROOT) if "==" in p], lock_text)
    # the environment folder is keyed on the lock digest, so a second Run all reuses it
    assert "'dimer_isolated_env_' + LOCK_SHA256[:12]" in source


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker, with every variable later cells created, is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("import os; print('value', learner_value, os.environ.get('MPLBACKEND'), os.environ.get('PYTHONSTARTUP'))\n")
        assert "value 42 Agg None" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- SWP-G: the guided layer ----------------------------------------------------------------------------------------


def test_swp_g_guided_layer_is_present_and_infrastructure_is_collapsed(notebook):
    md = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**How to use this notebook.**", "**Roadmap:**", "**Input → Model → Output.**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert heading in md, heading
    assert md.count("**Predict") >= 6
    assert md.count("<details><summary>Check your reasoning</summary>") >= 7
    assert md.index("**How to use this notebook.**") < md.index("## 1. Install the pinned runtime")
    for leftover in ("{{", "}}", "{MODEL_ID}", "@P:", "TODO", "TBD"):
        assert leftover not in md, leftover
    infra = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in _source(c) or c["metadata"].get("dimer", {}).get("embedded_module") or "stage_missing_files(WEIGHTS_DIR, allow_download=True)" in _source(c)]
    assert infra and all(c["metadata"].get("cellView") == "form" for c in infra)
    assert "> **Infrastructure.**" in md


def test_swp_g_checkpoint_answers_quote_the_recorded_run(notebook):
    md = _markdown(notebook)
    for value in ("0.1625", "6.8937", "0.1893", "0.8614", "0.1649", "0.1446", "0.0987"):
        assert value in md, value
