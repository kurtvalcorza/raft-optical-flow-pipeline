"""Static release-asset validation for the RAFT-Large (C_T_SKHT_V2) optical-flow DIMER pipeline.

Checks the STANDALONE tutorial notebook (DIMER Notebook Specification 2.2 §4), the tutorial
registry, model card (DIMER Model Card Specification 1.2), README, STATUS.md and weight documentation
for source conformance and cross-document identity consistency, the checkpoint pin state, and runs the
generator parity checks (PAR1–PAR3).

This is source validation only. A PASS here is NOT clean-runtime execution evidence;
the release gate is defined in docs/release-verification.md.
"""

# ruff: noqa: E501  -- rule messages name the file and requirement in full; they are kept on one line
from __future__ import annotations

import ast
import importlib.util
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "raft_optical_flow_pipeline"
REPO_NAME = "raft-optical-flow-pipeline"
NOTEBOOK_NAME = "raft_optical_flow_colab.ipynb"
EXPECTED_PROFILE = "E2E"
EXPECTED_MODEL_ID = "torchvision/raft_large"
WEIGHTS_KEY = "raft-large-c-t-skht-v2"
PIPELINE_CLASS = "RaftPipeline"
UNPINNED = "unpinned"
UNPINNED_PHRASE = "not yet pinned"
# Extra 40-hex commits the docs may legitimately cite (none yet).
KNOWN_SHAS: frozenset[str] = frozenset(())
# The checkpoint is a torchvision file on download.pytorch.org, not a Hugging Face Hub snapshot: its pinned
# revision is the SHA-256 of its bytes, and the notebook's external access is that host only.
WEIGHTS_URL = "https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
EXTERNAL_ACCESS_MARKER = "- **External access:** `download.pytorch.org` only"
REFERENCE_LINK = (
    "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html"
)

# NOTEBOOK_SPEC 2.2 §10.3: BYOD is gated off by default so the sample path runs top-to-bottom.
BYOD_GATES = ("USE_BYOD_IMAGE", "USE_BYOD_DATASET")
# NOTEBOOK_SPEC 2.2 EXE2: every file-reading branch has a location field.
LOCATION_FIELDS = ("BYOD_IMAGE1_PATH", "BYOD_IMAGE2_PATH", "BYOD_DATASET_DIR")

EXPECTED_OUTPUTS = (
    "raft_optical_flow_input_manifest.json",
    "raft_optical_flow_evaluation_report.json",
    "raft_optical_flow_result.json",
    "raft_optical_flow_metrics.csv",
    "raft_optical_flow_flow.flo",
    "raft_optical_flow_flow.npy",
    "raft_optical_flow_flow.png",
    "raft_adapter.safetensors",
)

CODE_MARKERS = (
    "demo = moving_shapes_pair(DEMO_SEED)",
    "input_manifest = validate_inputs(demo['image1'], demo['image2'], num_flow_updates=NUM_FLOW_UPDATES)",
    "demo_result = pipe.estimate(demo['image1'], demo['image2'], num_flow_updates=NUM_FLOW_UPDATES)",
    "demo_report = evaluation_report(demo_result, demo['flow'], demo['valid'], sample_kind='synthetic')",
    "NUM_FLOW_UPDATES = 12",
    "for updates in (1, 4, 12, 32):",
    "for probe in (static_pair(), blank_pair()):",
    "records = flow_dataset(N_PAIRS, seed=DATASET_SEED)",
    "dataset_manifest = validate_dataset(records, epochs=EPOCHS)",
    "train_records, held_out = split_pairs(records, train_fraction=1.0 - HOLDOUT, seed=SEED)",
    "adapter = RaftPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)",
    "baseline = adapter.evaluate(held_out, num_flow_updates=NUM_FLOW_UPDATES)",
    "run = adapter.finetune(",
    "freeze_encoders=FREEZE_ENCODERS,",
    "adapted = adapter.evaluate(held_out, num_flow_updates=NUM_FLOW_UPDATES)",
    "new_records = flow_dataset(3, seed=NEW_DATA_SEED)",
    "descriptor = adapter.save_artifact(artifact_path, notes='RAFT-Large moving-shapes adaptation tutorial adapter')",
    "reloaded = RaftPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)",
    "assert difference <= TOLERANCE_PX",
    "assert np.array_equal(read_flo(OUTPUTS / 'raft_optical_flow_flow.flo'), demo_result['flow'])",
    "byod_records = read_flow_records(dataset_dir)",
    "byod_pipe.finetune(byod_train",
    "'model_revision': MODEL_REVISION",
    "'model_license': MODEL_LICENSE",
    "torchvision.__version__",
    "'weights_url': WEIGHTS_URL",
    "'device': pipe.device",
)

MARKDOWN_MARKERS = (
    "**Capability:** dense optical flow between two frames",
    "**The default path really adapts the model:**",
    "**zero-flow baseline**",
    "**The pretrained model already beats the zero-flow baseline here.**",
    "**The feature and context encoders are frozen by default.**",
    "**Every BatchNorm layer is held in evaluation mode**",
    "**Read the loss as optimisation evidence only.**",
    "One pair gives the verdict `sample-sanity`, not a benchmark.",
    "Pairs cut from the same video must be split by video instead",
)

# Runtime/model-library access must stay inside the carried module (ST1/ST2).
FORBIDDEN_OUTSIDE_MODULE = (
    "raft_large(",
    "download_url_to_file(",
    "torch.load(",
    "load_state_dict(",
    "from safetensors",
    "from torchvision import",
    "torch.inference_mode(",
)

# The one kernel cell (generator /2.2 isolated runtime): it builds the hash-locked environment and routes every later
# cell to it, so it is the one place `urllib.request` belongs.
INSTALL_CELL_MARKER = "# dimer: kernel cell"

# ---------------------------------------------------------------------------
# Shared checks. Everything below is source/structure validation only. Passing
# these checks is NOT clean-runtime execution evidence under DIMER Notebook
# Specification 2.2; see docs/release-verification.md for the release gate.
# ---------------------------------------------------------------------------

NOTEBOOK_SPEC = "2.2"
MODEL_CARD_SPEC = "1.2"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_NAMES = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
# MODEL_CARD_SPEC 1.2 G14/G15/G17: a public card names no non-public capability, private source or
# internal build environment, and avoids fleet-maintenance vocabulary. These patterns catch the known
# forms; they do not replace the §6 review.
CARD_NON_PUBLIC = re.compile(
    r"ml-worker|workbench|internal (?:enterprise|tooling|platform)|\.venv\b|Windows venv|holdout surface",
    re.I,
)
CARD_FLEET_VOCABULARY = re.compile(r"\bfleet\b|\bwave\b|build queue|matrix row|inventory row", re.I)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
# Markers every standalone DIMER tutorial in this fleet must carry, independent of profile.
# Matched on comment-stripped code, so a commented-out call does not count.
COMMON_CODE_MARKERS = (
    "PINS = [",
    "NOTEBOOK_SOURCE = {",
    "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'",
    # SWP-R (2026-10-05 fleet sweep): the generator /2.2 isolated runtime replaces the in-kernel pinned install.
    "'--require-hashes', '--only-binary', ':all:'",
    "'--managed-python'",
    "if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:",
    "if hashlib.sha256(LOCK_TEXT.encode('utf-8')).hexdigest() != LOCK_SHA256:",
    "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)",
    "platform.python_version()",
    "torch.__version__",
    "MANIFEST = {",
    "if (MANIFEST['modelId'], MANIFEST['revision']) != (MODEL_ID, MODEL_REVISION):",
    "WEIGHTS_DIR = DEFAULT_WEIGHTS_DIR",
    "json.dump(MANIFEST, handle, indent=2)",
    "fetched = stage_missing_files(WEIGHTS_DIR, allow_download=True)",
    "snapshot = verify_snapshot(WEIGHTS_DIR)",
    "'repository_revision': NOTEBOOK_SOURCE['repository_revision']",
    "'notebook_source': NOTEBOOK_SOURCE",
    "os.makedirs('outputs', exist_ok=True)",
    "from google.colab import files",
    "files.upload()",
)
COMMON_MARKDOWN_MARKERS = (
    f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)",
    "**Mode:** `",
    "**Run all:**",
    "**Bring Your Own Data:**",
    "**This notebook is standalone.**",
    "**Learning objectives:**",
    "## Prerequisites",
    "Do not upload confidential or restricted",
    EXTERNAL_ACCESS_MARKER,
    "## 1. Install the pinned runtime",
    "## 2. Pipeline code (carried verbatim from",
    "## 3. Pin, stage and verify the model",
    "## Interpretation and limits",
    "Successful execution proves that the recorded repository revision",
    "without the repository being",
    "It does **not** establish benchmark superiority",
    "## References",
    f"- Repository model card: https://github.com/kurtvalcorza/{REPO_NAME}/blob/main/MODEL_CARD.md",
)
# Patterns that must never appear in tutorial code (comment-stripped), in any cell.
FORBIDDEN_PATTERNS = (
    ("credential in clone URL", re.compile(r"https://[^/'\"\s]*@github\.com/|x-access-token:")),
    ("repository clone (ST1)", re.compile(r"\bgit\b[^\n]*\bclone\b|github\.com")),
    ("editable self-install", re.compile(r"""['"](?:-e|--editable)['"]|pip install (?:-e|--editable)\b""")),
    ("repository package import (ST1)", re.compile(rf"^\s*(?:from|import)\s+{PACKAGE}\b", re.M)),
    ("mutable model reference (MOD14)", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    ("trust_remote_code enabled", re.compile(r"trust_remote_code\s*[=:]\s*True")),
    (
        "unsafe deserialization",
        re.compile(
            r"\bpickle\.load"
            r"|\btorch\.load\s*\((?![^)]*weights_only\s*=\s*True)"
            r"|weights_only\s*=\s*False"
            r"|getattr\(\s*torch\s*,\s*['\"]load['\"]"
        ),
    ),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
)


class ValidationError(AssertionError):
    """Raised for any release-asset defect; the message names the file and rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _strip_comments(source: str) -> str:
    """Return the source without comment tokens (string contents are preserved)."""
    out: list[str] = []
    last_row, last_col = 1, 0
    lines = source.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source
    for token in tokens:
        (srow, scol), (erow, ecol) = token.start, token.end
        if srow > last_row:
            out.append(lines[last_row - 1][last_col:] if last_row - 1 < len(lines) else "")
            for row in range(last_row, srow - 1):
                out.append(lines[row])
            last_row, last_col = srow, 0
        if srow - 1 < len(lines):
            out.append(lines[srow - 1][last_col:scol])
        if token.type != tokenize.COMMENT:
            out.append(token.string)
        last_row, last_col = erow, ecol
    return "".join(out)


def _assignment_targets(node: ast.AST):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr | ast.For | ast.comprehension):
        targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [node.optional_vars]
    else:
        return []
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    _check(spec is not None and spec.loader is not None, f"tools/{name}.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _package_identity() -> tuple[str, str]:
    """Read MODEL_ID / MODEL_REVISION from the package source without importing torch."""
    text = _read(ROOT / "src" / PACKAGE / "pipeline.py")
    model_id = re.search(r'^MODEL_ID = "([^"]+)"$', text, re.M)
    revision = re.search(r'^MODEL_REVISION = "([^"]+)"$', text, re.M)
    _check(
        model_id is not None and revision is not None,
        "pipeline.py must define MODEL_ID and MODEL_REVISION",
    )
    _check(
        SHA64.match(revision.group(1)) is not None or revision.group(1) == UNPINNED,
        f"MODEL_REVISION must be the checkpoint's 64-hex SHA-256 or the {UNPINNED!r} sentinel",
    )
    _check(model_id.group(1) == EXPECTED_MODEL_ID, f"MODEL_ID drifted from {EXPECTED_MODEL_ID}")
    return model_id.group(1), revision.group(1)


def _front_matter(text: str) -> dict[str, str]:
    front = text.split("---", 2)[1]
    fields: dict[str, str] = {}
    for line in front.splitlines():
        match = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if match:
            fields[match.group(1)] = match.group(2).strip().strip('"')
    return fields


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter (G1)")
    fields = _front_matter(text)
    for key in ("license", "model_card_spec", "pipeline_tag", "base_model", "date_published"):
        _check(key in fields, f"MODEL_CARD.md missing front-matter field: {key} (G1)")
    _check(
        fields["model_card_spec"] == MODEL_CARD_SPEC,
        f"MODEL_CARD.md model_card_spec must be {MODEL_CARD_SPEC!r} (1.0 is withdrawn, 1.1 is deprecated)",
    )
    _check(fields["base_model"] == EXPECTED_MODEL_ID, "MODEL_CARD.md base_model must equal MODEL_ID")
    _check(
        re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?|null", fields["date_published"]) is not None,
        "MODEL_CARD.md date_published must be YYYY, YYYY-MM, YYYY-MM-DD or null (G2)",
    )
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text (G9, G11)")
    _check(
        not UNSUPPORTED_CLAIMS.search(text),
        "MODEL_CARD.md makes an unsupported release/benchmark claim (G12)",
    )
    _check(
        not CARD_NON_PUBLIC.search(text),
        f"MODEL_CARD.md names a non-public capability or source (G14, G15): {CARD_NON_PUBLIC.search(text)}",
    )
    _check(
        not CARD_FLEET_VOCABULARY.search(text),
        f"MODEL_CARD.md uses fleet-maintenance vocabulary (G17): {CARD_FLEET_VOCABULARY.search(text)}",
    )
    preamble = text.split("#### Description", 1)[0]
    badge_targets = re.findall(r"\[!\[[^\]]*\]\([^)]*\)\]\(([^)]+)\)", preamble)
    self_links = [t for t in badge_targets if t.rstrip("/") == f"https://github.com/kurtvalcorza/{REPO_NAME}"]
    _check(not self_links, "MODEL_CARD.md badge row must not link back to this repository (G8)")
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)} (G3)")
    found = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [
            index
            for index, item in enumerate(found)
            if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()
        ]
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading} (G4, G5)")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order (G6)")
    _check("## Immutable provenance" in text, "MODEL_CARD.md must carry an '## Immutable provenance' section")


def _manifest() -> dict:
    return json.loads(_read(ROOT / "weights" / WEIGHTS_KEY / "dimer-base-manifest.json"))


def validate_pin_state() -> None:
    """The package, the manifest and the documents agree on whether the checkpoint is pinned."""
    model_id, revision = _package_identity()
    manifest = _manifest()
    _check(
        manifest.get("format") == "dimer_url_snapshot",
        "manifest format must be dimer_url_snapshot (a URL-hosted checkpoint)",
    )
    _check(manifest.get("modelId") == model_id, "manifest modelId != MODEL_ID")
    _check(
        manifest.get("revision") == revision,
        f"manifest revision {manifest.get('revision')!r} != MODEL_REVISION {revision!r}",
    )
    _check(
        len(manifest["files"]) == 1 and manifest["files"][0].get("url") == WEIGHTS_URL,
        f"manifest must list exactly the one checkpoint at {WEIGHTS_URL}",
    )
    [entry] = manifest["files"]
    docs = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md", "STATUS.md")
    if revision == UNPINNED:
        _check(
            entry.get("sha256") is None and entry.get("bytes") is None and manifest.get("totalBytes") is None,
            "an unpinned manifest must not carry a digest or a size; run tools/pin_snapshot.py to record both from the downloaded file",
        )
        for name in docs:
            _check(
                UNPINNED_PHRASE in _read(ROOT / name),
                f"{name} must state that the snapshot is {UNPINNED_PHRASE} while MODEL_REVISION is {UNPINNED!r}",
            )
        status = _read(ROOT / "STATUS.md")
        _check("Current status: **Candidate" in status, "an unpinned snapshot can only be Candidate")
    else:
        _check(
            entry.get("sha256") == revision,
            "a pinned manifest must record the checkpoint SHA-256, and it must equal MODEL_REVISION",
        )
        _check(
            revision.startswith(WEIGHTS_URL.rsplit("-", 1)[1].split(".")[0]),
            "the pinned SHA-256 must start with the prefix in the torchvision file name",
        )
        _check(
            isinstance(entry.get("bytes"), int) and manifest.get("totalBytes") == entry["bytes"],
            "a pinned manifest must record the checkpoint byte size as bytes and totalBytes",
        )
        for name in docs + ("tutorials/README.md", "docs/release-verification.md"):
            _check(
                UNPINNED_PHRASE not in _read(ROOT / name),
                f"{name} still says the snapshot is {UNPINNED_PHRASE}; revise it for the pinned commit",
            )


def validate_identity_consistency() -> None:
    """The upstream identity must be the same string in every document."""
    model_id, revision = _package_identity()
    for name in ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md"):
        text = _read(ROOT / name)
        _check(model_id in text, f"{name} must name the upstream model `{model_id}`")
        _check(revision in text, f"{name} must cite the revision {revision}")
        other = re.findall(r"\b[0-9a-f]{40}\b", text)
        stray = sorted({sha for sha in other if sha != revision and sha not in KNOWN_SHAS})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")


# --- weight-facts check (fleet rollout 2026-09-24) ---
# Every SHA-256 digest and byte count quoted in the weight prose must come from a committed
# weights/*/dimer-base-manifest.json, or be declared below with a label saying what it describes
# (dataset files, upstream files that are not staged, origin checkpoints, totals). Declared entries
# that no document cites any more are rejected, so the allowlist cannot go stale.
WEIGHT_DOCS = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md")
EXTERNAL_WEIGHT_BYTES: dict[int, str] = {}
EXTERNAL_WEIGHT_DIGESTS: dict[str, str] = {}
_DIGEST = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
_GROUPED = r"(\d{1,3}(?:[,   ]\d{3})+|\d+)"
_BYTE_COUNT = re.compile(r"(?<![\d,\-])" + _GROUPED + r"\s*bytes\b|totalBytes`?\s*" + _GROUPED)


def _manifest_facts(root: Path = ROOT) -> tuple[set[str], set[int]]:
    digests: set[str] = set()
    sizes: set[int] = set()
    for path in sorted(root.glob("weights/*/dimer-base-manifest.json")):
        manifest = json.loads(_read(path))
        if manifest.get("totalBytes") is not None:
            sizes.add(manifest["totalBytes"])
        for entry in manifest["files"]:
            if entry.get("sha256"):
                digests.add(entry["sha256"])
            if entry.get("bytes") is not None:
                sizes.add(entry["bytes"])
    return digests, sizes


def validate_weight_facts(root: Path = ROOT) -> None:
    """Every SHA-256 and byte count quoted in the weight prose must come from a manifest or a labelled allowlist entry."""
    digests, sizes = _manifest_facts(root)
    _check(
        bool(list(root.glob("weights/*/dimer-base-manifest.json"))),
        "no weights/*/dimer-base-manifest.json found to check weight facts against",
    )
    cited_digests: set[str] = set()
    cited_sizes: set[int] = set()
    for name in WEIGHT_DOCS:
        path = root / name
        if not path.exists():
            continue
        text = _read(path)
        found_digests = set(_DIGEST.findall(text))
        found_sizes = {
            int(re.sub(r"[,   ]", "", m.group(1) or m.group(2))) for m in _BYTE_COUNT.finditer(text)
        }
        cited_digests |= found_digests
        cited_sizes |= found_sizes
        bad_digests = sorted(found_digests - digests - set(EXTERNAL_WEIGHT_DIGESTS))
        _check(
            not bad_digests,
            f"{name} cites SHA-256 digests absent from every manifest and from EXTERNAL_WEIGHT_DIGESTS: {bad_digests}",
        )
        bad_sizes = sorted(found_sizes - sizes - set(EXTERNAL_WEIGHT_BYTES))
        _check(
            not bad_sizes,
            f"{name} cites byte counts absent from every manifest and from EXTERNAL_WEIGHT_BYTES: {bad_sizes}",
        )
    stale = sorted(set(EXTERNAL_WEIGHT_BYTES) - cited_sizes) + sorted(
        set(EXTERNAL_WEIGHT_DIGESTS) - cited_digests
    )
    _check(not stale, f"EXTERNAL_WEIGHT_* entries no weight document cites any more: {stale}")


# --- end weight-facts check ---


def validate_release_status() -> None:
    """STATUS.md, README.md and tutorials/README.md must agree on one status token."""
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade)\b", status)
    _check(match is not None, "STATUS.md must declare 'Current status: **Candidate**' or '**Release-grade**'")
    token = match.group(1)
    readme = _read(ROOT / "README.md")
    _check("## Release status" in readme, "README.md must have a '## Release status' section")
    section = readme.split("## Release status", 1)[1]
    _check(section.lstrip().startswith(f"**{token}"), f"README.md release status must open with **{token}**")
    registry = _read(ROOT / "tutorials" / "README.md").replace("**", "")
    _check(f"| {token}" in registry, f"tutorials/README.md must record the {token} status")
    other = [t for t in STATUS_TOKENS if t != token]
    for name, text in (("README.md", section.replace("**", "")), ("tutorials/README.md", registry)):
        for stale in other:
            _check(f"| {stale}" not in text, f"{name} carries a conflicting status token")
    if token == "Candidate":
        _check(
            "docs/release-verification.md" in registry or "release-verification" in registry,
            "tutorials/README.md must point Candidate notebooks at docs/release-verification.md",
        )
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md"):
        text = _read(ROOT / name)
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/benchmark claim")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md must have '## Recorded executions'",
    )


def _validate_notebook_structure(path: Path, notebook: dict) -> tuple[list[tuple[int, str, ast.Module]], str]:
    _check(notebook.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer")
    _check(isinstance(dimer, dict), f"{path.name}: metadata.dimer block is required")
    profile = dimer.get("notebook_profile")
    _check(profile in ALLOWED_PROFILES, f"{path.name}: invalid metadata.dimer.notebook_profile {profile!r}")
    _check(profile == EXPECTED_PROFILE, f"{path.name}: profile {profile!r} != declared {EXPECTED_PROFILE!r}")
    spec = dimer.get("notebook_spec", dimer.get("notebook_spec_version"))
    _check(
        spec == NOTEBOOK_SPEC,
        f"{path.name}: metadata.dimer must declare notebook spec version '{NOTEBOOK_SPEC}'",
    )
    _check(
        dimer.get("notebook_mode") in ("REFERENCE", "GUIDED", "WORKSHOP"),
        f"{path.name}: metadata.dimer.notebook_mode must declare a §3.3 pedagogical mode",
    )
    _check(dimer.get("standalone") is True, f"{path.name}: metadata.dimer.standalone must be true (ST6)")
    generated = dimer.get("generated_from")
    _check(isinstance(generated, dict), f"{path.name}: metadata.dimer.generated_from is required (ST5)")
    _check(
        generated.get("repository") == REPO_NAME,
        f"{path.name}: generated_from.repository must be {REPO_NAME}",
    )
    _check(
        generated.get("module") == f"src/{PACKAGE}/pipeline.py",
        f"{path.name}: generated_from.module must be src/{PACKAGE}/pipeline.py",
    )
    template = _load_tool("notebook_template").TEMPLATE
    build_tool = _load_tool("build_notebook")
    context = build_tool.load_context(ROOT, template)
    _check(
        generated.get("module_sha256") == context["module_sha256"],
        f"{path.name}: generated_from.module_sha256 does not match src/ (PAR4: regenerate the notebook)",
    )
    _check(
        generated.get("modules") == [f"src/{PACKAGE}/{m}" for m in context["modules"]],
        f"{path.name}: generated_from.modules must list every carried module in dependency order",
    )
    _check(bool(generated.get("generator")), f"{path.name}: generated_from.generator is required")
    cells = notebook.get("cells", [])
    _check(
        bool(cells) and cells[0].get("cell_type") == "markdown",
        f"{path.name}: first cell must be markdown",
    )
    code_cells: list[tuple[int, str, ast.Module]] = []
    markdown_parts: list[str] = []
    for index, cell in enumerate(cells):
        source = _cell_source(cell)
        if cell.get("cell_type") == "markdown":
            markdown_parts.append(source)
            continue
        _check(cell.get("cell_type") == "code", f"{path.name}: unexpected cell type at {index}")
        _check(cell.get("execution_count") is None, f"{path.name}: code cell {index} has execution_count")
        _check(not cell.get("outputs"), f"{path.name}: code cell {index} persists outputs")
        _check(
            index > 0 and cells[index - 1].get("cell_type") == "markdown",
            f"{path.name}: code cell {index} lacks a preceding explanatory markdown cell",
        )
        for line in source.splitlines():
            _check(not line.lstrip().startswith(("%", "!")), f"{path.name}: cell {index} uses a magic")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"{path.name}: code cell {index} does not compile: {exc}") from exc
        code_cells.append((index, source, tree))
    markdown = "\n".join(markdown_parts)
    raw_code = "\n".join(source for _, source, _ in code_cells)
    _check(not PLACEHOLDER.search(raw_code + markdown), f"{path.name}: placeholder text found")
    _check(not UNSUPPORTED_CLAIMS.search(markdown), f"{path.name}: unsupported release/benchmark claim")
    return code_cells, markdown


def _form_assignments(
    code_cells: list[tuple[int, str, ast.Module]], name: str
) -> list[tuple[int, ast.AST, str]]:
    found = []
    for index, source, tree in code_cells:
        lines = source.splitlines()
        for node in ast.walk(tree):
            if name in _assignment_targets(node):
                line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
                found.append((index, node, line))
    return found


def _validate_gates(path: Path, code_cells: list[tuple[int, str, ast.Module]]) -> None:
    """Each BYOD gate is assigned exactly once, to the constant False, on a Colab form line; each location
    field is an empty-string form field (EXE1, EXE2)."""
    for gate in BYOD_GATES:
        assignments = _form_assignments(code_cells, gate)
        _check(
            len(assignments) == 1,
            f"{path.name}: {gate} must be assigned exactly once, found {len(assignments)}",
        )
        index, node, line = assignments[0]
        is_false = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        )
        _check(is_false, f"{path.name}: {gate} must be assigned the constant False (cell {index})")
        _check("# @param" in line, f"{path.name}: {gate} must be a Colab form parameter (`# @param`)")
    for field_name in LOCATION_FIELDS:
        assignments = _form_assignments(code_cells, field_name)
        _check(
            len(assignments) == 1,
            f"{path.name}: location field {field_name} must be assigned exactly once (EXE2)",
        )
        index, node, line = assignments[0]
        is_empty = (
            isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value == ""
        )
        _check(
            is_empty and "# @param" in line,
            f"{path.name}: {field_name} must be an empty-string `# @param` form field (EXE1, cell {index})",
        )
    for index, _source, tree in code_cells:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                _check(
                    not any(alias.name.startswith("google.colab") for alias in node.names),
                    f"{path.name}: google.colab must only be imported inside the BYOD gate (cell {index})",
                )


def _validate_embedded_module(path: Path, notebook: dict, build) -> set[int]:
    """PAR1: one tagged cell per carried module, each equal to its module after the rewrites."""
    template = _load_tool("notebook_template").TEMPLATE
    modules = build.load_context(ROOT, template)["modules"]
    rewrites = template.get("rewrites", build.DEFAULT_REWRITES)
    tagged = [
        (index, cell)
        for index, cell in enumerate(notebook.get("cells", []))
        if cell.get("cell_type") == "code"
        and cell.get("metadata", {}).get("dimer", {}).get("embedded_module")
    ]
    _check(
        len(tagged) == len(modules),
        f"{path.name}: expected {len(modules)} cells tagged metadata.dimer.embedded_module, found {len(tagged)} (ST2)",
    )
    names = [cell["metadata"]["dimer"]["embedded_module"] for _index, cell in tagged]
    _check(
        names == [f"src/{PACKAGE}/{m}" for m in modules],
        f"{path.name}: embedded_module tags {names} do not match the carried modules in dependency order",
    )
    texts = {m: _read(ROOT / "src" / PACKAGE / m) for m in modules}
    expected = build.apply_rewrites(texts, rewrites)
    for (index, cell), module in zip(tagged, modules, strict=True):
        _check(
            _cell_source(cell).rstrip("\n") + "\n" == expected[module],
            f"{path.name}: embedded module differs from src/{PACKAGE}/{module} (PAR1); regenerate the notebook (cell {index})",
        )
    return {index for index, _cell in tagged}


def _validate_identity(
    path: Path, code_cells: list[tuple[int, str, ast.Module]], embedded_indices: set[int], revision: str
) -> None:
    """Identity constants are bound in the carried modules only; nothing outside rebinds them."""
    for index, _source, tree in code_cells:
        if index in embedded_indices:
            continue
        for node in ast.walk(tree):
            rebound = [name for name in _assignment_targets(node) if name in IDENTITY_NAMES]
            _check(
                not rebound,
                f"{path.name}: {rebound} must not be rebound outside the module cell (cell {index})",
            )
    outside = "\n".join(source for index, source, _ in code_cells if index not in embedded_indices)
    manifest_block = re.search(r"^MANIFEST = (\{.*?^\})$", outside, re.M | re.S)
    _check(manifest_block is not None, f"{path.name}: model cell must carry an inline MANIFEST literal (ST3)")
    if revision != UNPINNED:
        outside_without_manifest = outside.replace(manifest_block.group(0), "")
        _check(
            revision not in outside_without_manifest,
            f"{path.name}: the model revision may appear only in the carried module and the inline manifest",
        )


def _validate_parity(
    path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], build
) -> None:
    """PAR2/PAR3: inline manifest and pins equal the repository's; the generator reproduces the file."""
    template = _load_tool("notebook_template").TEMPLATE
    code = "\n".join(source for _, source, _ in code_cells)
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.M | re.S)
    _check(
        inline is not None and json.loads(inline.group(1)) == _manifest(),
        f"{path.name}: inline MANIFEST != committed manifest (PAR2)",
    )
    pins_block = re.search(r"^PINS = \[(.*?)^\]", code, re.M | re.S)
    _check(pins_block is not None, f"{path.name}: install cell must carry PINS = [...] (ENV2)")
    _check(
        re.findall(r"'([^']+)'", pins_block.group(1)) == build._pins(ROOT),
        f"{path.name}: inline PINS != pyproject runtime pins (PAR2)",
    )
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, template, recorded))
    current = path.read_bytes().replace(b"\r\n", b"\n")  # autocrlf checkouts are CRLF
    _check(
        current == rendered, f"{path.name}: differs from tools/build_notebook.py output (PAR3); regenerate"
    )


def _validate_bootstrap_guard(path: Path, code_cells: list[tuple[int, str, ast.Module]]) -> None:
    """RUN1/RUN10/ENV6 (SWP-R, 2026-10-05 fleet sweep): nothing is pip-installed into the kernel and no cell asks for a
    restart. Exactly one cell runs in the kernel (the isolated-environment bootstrap); it reuses a matching environment."""
    kernel = [source for _, source, _ in code_cells if INSTALL_CELL_MARKER in source]
    _check(len(kernel) == 1, f"{path.name}: exactly one '{INSTALL_CELL_MARKER}' bootstrap cell is required, found {len(kernel)}")
    code = "\n".join(source for _, source, _ in code_cells)
    _check("'-m', 'pip', 'install'" not in code and "pip install" not in code, f"{path.name}: no cell may pip-install into the notebook kernel (RUN10)")
    _check("Restart the runtime" not in code, f"{path.name}: no cell may ask for a runtime restart (RUN1)")
    _check("_isolated_environment_ready()" in kernel[0], f"{path.name}: the bootstrap cell must reuse a matching isolated environment")

def _validate_notebook_content(
    path: Path, code_cells: list[tuple[int, str, ast.Module]], markdown: str, embedded_indices: set[int]
) -> None:
    model_id, _revision = _package_identity()
    stripped = {index: _strip_comments(source) for index, source, _ in code_cells}
    code = "\n".join(stripped.values())
    outside = "\n".join(text for index, text in stripped.items() if index not in embedded_indices)
    missing = [marker for marker in COMMON_CODE_MARKERS + CODE_MARKERS if marker not in code]
    _check(not missing, f"{path.name}: missing required source markers: {missing}")
    present = [label for label, pattern in FORBIDDEN_PATTERNS if pattern.search(code)]
    _check(not present, f"{path.name}: forbidden/insecure source: {present}")
    outside_stage_cells = "\n".join(
        text for index, text in stripped.items() if index not in embedded_indices and INSTALL_CELL_MARKER not in text
    )
    leaked = [marker for marker in FORBIDDEN_OUTSIDE_MODULE if marker in outside_stage_cells]
    _check(not leaked, f"{path.name}: direct library use outside the carried module cell (G2): {leaked}")
    _check(
        f"pipe = {PIPELINE_CLASS}.from_pretrained(weights_dir=WEIGHTS_DIR)" in outside,
        f"{path.name}: must load through {PIPELINE_CLASS}.from_pretrained(weights_dir=WEIGHTS_DIR) (INF1)",
    )
    _validate_gates(path, code_cells)
    _validate_bootstrap_guard(path, code_cells)
    for filename in EXPECTED_OUTPUTS:
        _check(filename in code, f"{path.name}: must export {filename}")
    missing_md = [marker for marker in COMMON_MARKDOWN_MARKERS + MARKDOWN_MARKERS if marker not in markdown]
    _check(not missing_md, f"{path.name}: missing learner-facing markers: {missing_md}")
    _check(f"**Profile:** `{EXPECTED_PROFILE}`" in markdown, f"{path.name}: markdown must state the profile")
    _check(
        REFERENCE_LINK in markdown,
        f"{path.name}: references must link the torchvision documentation for {model_id}",
    )


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    notebooks = sorted(tutorials.glob("*.ipynb"))
    _check(len(notebooks) == 1, f"exactly one tutorial notebook is expected, found {len(notebooks)}")
    path = notebooks[0]
    _check(path.name == NOTEBOOK_NAME, f"tutorial notebook must be named {NOTEBOOK_NAME}, found {path.name}")
    build = _load_tool("build_notebook")
    _check(
        build.NOTEBOOK_SPEC == NOTEBOOK_SPEC,
        f"tools/build_notebook.py targets notebook spec {build.NOTEBOOK_SPEC}, expected {NOTEBOOK_SPEC}",
    )
    notebook = json.loads(_read(path))
    code_cells, markdown = _validate_notebook_structure(path, notebook)
    embedded_indices = _validate_embedded_module(path, notebook, build)
    _model_id, revision = _package_identity()
    _validate_identity(path, code_cells, embedded_indices, revision)
    _validate_parity(path, notebook, code_cells, build)
    _validate_notebook_content(path, code_cells, markdown, embedded_indices)
    registry = _read(tutorials / "README.md")
    _check(f"`{path.name}`" in registry, f"{path.name} missing from tutorials/README.md")
    _check(f"`{EXPECTED_PROFILE}`" in registry, f"tutorials/README.md must record `{EXPECTED_PROFILE}`")
    _check(
        f"DIMER Notebook Specification {NOTEBOOK_SPEC}" in registry,
        "tutorials/README.md must name the notebook spec version",
    )
    _check(
        "standalone" in registry.lower(), "tutorials/README.md must record that the notebook is standalone"
    )


def validate_all() -> list[str]:
    validate_model_card()
    validate_pin_state()
    validate_identity_consistency()
    validate_weight_facts()
    validate_release_status()
    validate_notebooks()
    return [
        "model-card",
        "pin-state",
        "identity-consistency",
        "weight-facts",
        "release-status",
        "notebook+parity",
    ]


def main() -> int:
    passed = validate_all()
    print(f"release asset validation: PASS ({', '.join(passed)})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
