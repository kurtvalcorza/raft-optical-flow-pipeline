#!/usr/bin/env python3
"""Pin the upstream checkpoint: record the SHA-256 and byte size of the one torchvision weight file.

The weights are not on the Hugging Face Hub. torchvision publishes them at a fixed URL on
download.pytorch.org whose file name carries the first 8 hex digits of the file's SHA-256
(``...-ff5fadd5.pth``). The committed manifest lists that URL with no digest and no size (`"revision":
"unpinned"`, `"sha256": null`, `"bytes": null`). Until this tool has run, the package refuses to stage,
verify or load weights.

What it does, in order:

1. downloads the manifest URL into a staging file, replacing any local copy;
2. computes the file's SHA-256 and byte size, and stops without writing anything if the digest does not
   start with the 8 hex digits in the file name;
3. loads the file with ``torch.load(weights_only=True)`` into torchvision's ``raft_large``
   built with ``weights=None`` and ``load_state_dict(strict=True)``, and stops if any tensor is missing,
   unexpected or mis-shaped;
4. moves the file into ``weights/<key>/``, writes the manifest (revision = the full SHA-256, bytes,
   sha256, totalBytes) and replaces ``MODEL_REVISION = "unpinned"`` in ``src/<package>/pipeline.py``
   with the digest.

A URL-hosted file has no commit, so the pinned revision of this checkpoint is the SHA-256 of its bytes.
The tool then prints what is left to do by hand: regenerate the notebook, update the prose that says the
checkpoint is not yet pinned, and run the validator. Needs network access to download.pytorch.org and the
pinned torch and torchvision; it imports nothing from this repository's package.

Usage (from the repository root):
    python tools/pin_snapshot.py            # download, hash, load-check and pin
    python tools/pin_snapshot.py --dry-run  # download, hash and load-check; write nothing
"""

# ruff: noqa: E501  -- printed guidance is kept on one line per message
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = re.compile(r"-([0-9a-f]{8,})\.pth$")


def _template(root: Path) -> dict:
    spec = importlib.util.spec_from_file_location(
        "notebook_template", root / "tools" / "notebook_template.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TEMPLATE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    from torch.hub import download_url_to_file

    download_url_to_file(url, str(target), progress=True)


def _load_check(path: Path) -> str:
    """Strict-load the checkpoint into torchvision's architecture; return a one-line summary."""
    import torch
    from torchvision.models.optical_flow import raft_large

    model = raft_large(weights=None, progress=False)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return f"{len(state)} tensors, {sum(p.numel() for p in model.parameters()):,} parameters, strict load ok"


def pin(
    root: Path = ROOT,
    *,
    dry_run: bool = False,
    download: Callable[[str, Path], None] | None = None,
    load_check: Callable[[Path], str] | None = None,
) -> int:
    """Pin the checkpoint under ``root``; ``download``/``load_check`` default to torch.hub and torchvision."""
    download = download or _download
    load_check = load_check or _load_check

    template = _template(root)
    key = template["weights_key"]
    weights_dir = root / "weights" / key
    manifest_path = weights_dir / "dimer-base-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["files"]) != 1:
        print(
            f"expected one checkpoint file in {manifest_path}, found {len(manifest['files'])}",
            file=sys.stderr,
        )
        return 1
    [entry] = manifest["files"]
    match = PREFIX.search(entry["path"])
    if not match or not entry["url"].endswith("/" + entry["path"]):
        print(
            f"{entry['path']}: file name must end in -<sha256 prefix>.pth and match the URL {entry['url']}",
            file=sys.stderr,
        )
        return 1

    staging = root / "outputs" / "pin-staging" / key
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / entry["path"]
    if staged.exists():
        staged.unlink()
    print(f"downloading {entry['url']}")
    download(entry["url"], staged)
    size = staged.stat().st_size
    digest = _sha256(staged)
    if not digest.startswith(match.group(1)):
        print(
            f"{entry['path']}: sha256 {digest} does not start with the file-name prefix {match.group(1)}; nothing written",
            file=sys.stderr,
        )
        return 1
    try:
        summary = load_check(staged)
    except Exception as exc:  # noqa: BLE001 -- any load failure means the file must not be pinned
        print(
            f"{entry['path']}: strict load into raft_large failed ({exc}); nothing written",
            file=sys.stderr,
        )
        return 1
    print(f"  {entry['path']}: {size:,} bytes  sha256 {digest}  (prefix {match.group(1)} matches; {summary})")

    pinned = {
        **manifest,
        "revision": digest,
        "files": [{**entry, "bytes": size, "sha256": digest}],
        "totalBytes": size,
    }
    if dry_run:
        print(json.dumps(pinned, indent=2))
        return 0

    module_path = root / "src" / template["package"] / template.get("entry_module", "pipeline.py")
    text = module_path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'^MODEL_REVISION = "[^"]*"$', f'MODEL_REVISION = "{digest}"', text, count=1, flags=re.M
    )
    if n != 1:
        print(f"{module_path}: MODEL_REVISION constant not found; nothing written", file=sys.stderr)
        return 1
    shutil.move(str(staged), str(weights_dir / entry["path"]))
    manifest_path.write_text(json.dumps(pinned, indent=2) + "\n", encoding="utf-8", newline="\n")
    module_path.write_text(new_text, encoding="utf-8", newline="\n")
    print(f"wrote {manifest_path.relative_to(root)} and MODEL_REVISION in {module_path.relative_to(root)}")

    leftovers = [
        name
        for name in (
            "README.md",
            "MODEL_CARD.md",
            "STATUS.md",
            "docs/WEIGHTS.md",
            "tutorials/README.md",
            "docs/release-verification.md",
        )
        if (root / name).exists()
        and (
            "not yet pinned" in (root / name).read_text(encoding="utf-8")
            or "unpinned" in (root / name).read_text(encoding="utf-8")
        )
    ]
    print("next:")
    print("  1. commit, then run `python tools/build_notebook.py` and commit the regenerated notebook")
    if leftovers:
        print(
            f"  2. replace the 'not yet pinned' statements in: {', '.join(leftovers)} (cite the digest and byte size)"
        )
    print("  3. run `python tools/validate_release_assets.py` and `pytest`")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="download, hash and load-check only; write nothing"
    )
    args = parser.parse_args(argv)
    return pin(ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
