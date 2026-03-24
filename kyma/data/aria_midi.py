"""Download and local-cache helpers for the public Aria-MIDI dataset."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

from tqdm import tqdm

ARIA_MIDI_DATASET_CARD_URL = "https://huggingface.co/datasets/loubb/aria-midi"
ARIA_MIDI_REPO_RESOLVE_URL = (
    "https://huggingface.co/datasets/loubb/aria-midi/resolve/main"
)
ARIA_MIDI_LICENSE = "CC-BY-NC-SA 4.0"
ARIA_MIDI_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
ARIA_MIDI_PAPER_URL = "https://openreview.net/forum?id=X5hrhgndxW"
ARIA_MIDI_DEFAULT_ROOT = Path("artifacts/data/aria-midi")


@dataclass(frozen=True)
class AriaMidiSubsetSpec:
    """Static metadata for one published Aria-MIDI subset."""

    subset: str
    archive_filename: str
    recommended_use: str
    files: int
    size_label: str
    preprocess_filename: str | None = None


@dataclass(frozen=True)
class AriaMidiDownloadPlan:
    """Resolved local cache plan for one Aria-MIDI subset."""

    subset: str
    root: str
    archive_path: str
    archive_url: str
    preprocess_path: str | None
    preprocess_url: str | None
    disclaimer_path: str
    disclaimer_url: str
    readme_path: str
    readme_url: str
    manifest_path: str
    license_name: str
    license_url: str
    paper_url: str
    files: int
    size_label: str
    recommended_use: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ARIA_MIDI_SUBSETS: dict[str, AriaMidiSubsetSpec] = {
    "full": AriaMidiSubsetSpec(
        subset="full",
        archive_filename="aria-midi-v1-ext.tar.gz",
        recommended_use="Data analysis",
        files=1_186_253,
        size_label="8.51 GB",
    ),
    "pruned": AriaMidiSubsetSpec(
        subset="pruned",
        archive_filename="aria-midi-v1-pruned-ext.tar.gz",
        recommended_use="Foundation model pre-training",
        files=820_944,
        size_label="5.41 GB",
        preprocess_filename="light-preprocess.json",
    ),
    "deduped": AriaMidiSubsetSpec(
        subset="deduped",
        archive_filename="aria-midi-v1-deduped-ext.tar.gz",
        recommended_use="Generative modeling",
        files=371_053,
        size_label="2.01 GB",
        preprocess_filename="heavy-preprocess.json",
    ),
    "unique": AriaMidiSubsetSpec(
        subset="unique",
        archive_filename="aria-midi-v1-unique-ext.tar.gz",
        recommended_use="Composition fingerprints",
        files=32_522,
        size_label="254 MB",
    ),
}


def _resolve_url(filename: str) -> str:
    return f"{ARIA_MIDI_REPO_RESOLVE_URL}/{filename}?download=true"


def build_aria_midi_download_plan(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
) -> AriaMidiDownloadPlan:
    """Build a local-cache plan for one published Aria-MIDI subset."""

    if subset not in ARIA_MIDI_SUBSETS:
        choices = ", ".join(sorted(ARIA_MIDI_SUBSETS))
        raise ValueError(f"Unknown subset {subset!r}. Expected one of: {choices}.")

    spec = ARIA_MIDI_SUBSETS[subset]
    root_path = Path(root) / subset
    preprocess_path = None
    preprocess_url = None
    if spec.preprocess_filename is not None:
        preprocess_path = str(root_path / spec.preprocess_filename)
        preprocess_url = _resolve_url(spec.preprocess_filename)

    return AriaMidiDownloadPlan(
        subset=subset,
        root=str(root_path),
        archive_path=str(root_path / spec.archive_filename),
        archive_url=_resolve_url(spec.archive_filename),
        preprocess_path=preprocess_path,
        preprocess_url=preprocess_url,
        disclaimer_path=str(root_path / "DISCLAIMER.md"),
        disclaimer_url=_resolve_url("DISCLAIMER.md"),
        readme_path=str(root_path / "README.upstream.md"),
        readme_url=_resolve_url("README.md"),
        manifest_path=str(root_path / "manifest.json"),
        license_name=ARIA_MIDI_LICENSE,
        license_url=ARIA_MIDI_LICENSE_URL,
        paper_url=ARIA_MIDI_PAPER_URL,
        files=spec.files,
        size_label=spec.size_label,
        recommended_use=spec.recommended_use,
    )


def _download_to_path(
    url: str,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    existing_size = 0
    if destination_path.exists() and not overwrite:
        existing_size = int(destination_path.stat().st_size)
    elif overwrite and destination_path.exists():
        destination_path.unlink()

    headers: dict[str, str] = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    try:
        response = request.urlopen(request.Request(url, headers=headers))
    except error.HTTPError as exc:
        if exc.code == 416 and existing_size > 0 and not overwrite:
            return
        raise
    status = getattr(response, "status", 200)
    content_range = response.headers.get("Content-Range")
    content_length = response.headers.get("Content-Length")
    total_size: int | None = None
    if content_range:
        total_size = int(content_range.rsplit("/", 1)[1])
    elif content_length:
        current_length = int(content_length)
        total_size = existing_size + current_length if status == 206 else current_length

    mode = "ab" if existing_size > 0 and status == 206 else "wb"
    if mode == "wb":
        existing_size = 0

    with response, destination_path.open(mode) as handle:
        progress = tqdm(
            total=total_size,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination_path.name,
            disable=not sys.stderr.isatty(),
        )
        with progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))


def _manifest_payload(plan: AriaMidiDownloadPlan) -> dict[str, Any]:
    archive_path = Path(plan.archive_path)
    payload: dict[str, Any] = {
        "dataset": "aria-midi",
        "subset": plan.subset,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "dataset_card_url": ARIA_MIDI_DATASET_CARD_URL,
        "license": {
            "name": plan.license_name,
            "url": plan.license_url,
        },
        "paper_url": plan.paper_url,
        "recommended_use": plan.recommended_use,
        "files": plan.files,
        "size_label": plan.size_label,
        "paths": {
            "root": plan.root,
            "archive": plan.archive_path,
            "readme": plan.readme_path,
            "disclaimer": plan.disclaimer_path,
            "manifest": plan.manifest_path,
        },
        "source_urls": {
            "archive": plan.archive_url,
            "readme": plan.readme_url,
            "disclaimer": plan.disclaimer_url,
        },
    }
    if archive_path.exists():
        payload["archive_size_bytes"] = int(archive_path.stat().st_size)
    if plan.preprocess_path is not None and plan.preprocess_url is not None:
        payload["paths"]["preprocess"] = plan.preprocess_path
        payload["source_urls"]["preprocess"] = plan.preprocess_url
    return payload


def download_aria_midi(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
    accept_license: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download one published Aria-MIDI subset into the local cache."""

    plan = build_aria_midi_download_plan(subset=subset, root=root)
    if dry_run:
        return plan.to_dict()
    if not accept_license:
        raise ValueError(
            "Downloading Aria-MIDI requires explicit license acceptance. "
            "Pass accept_license=True after reviewing the dataset card, "
            "license, and disclaimer."
        )

    root_path = Path(plan.root)
    root_path.mkdir(parents=True, exist_ok=True)
    _download_to_path(plan.disclaimer_url, plan.disclaimer_path, overwrite=overwrite)
    _download_to_path(plan.readme_url, plan.readme_path, overwrite=overwrite)
    if plan.preprocess_url is not None and plan.preprocess_path is not None:
        _download_to_path(
            plan.preprocess_url,
            plan.preprocess_path,
            overwrite=overwrite,
        )
    _download_to_path(plan.archive_url, plan.archive_path, overwrite=overwrite)

    manifest = _manifest_payload(plan)
    Path(plan.manifest_path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "ARIA_MIDI_DATASET_CARD_URL",
    "ARIA_MIDI_DEFAULT_ROOT",
    "ARIA_MIDI_SUBSETS",
    "AriaMidiDownloadPlan",
    "AriaMidiSubsetSpec",
    "build_aria_midi_download_plan",
    "download_aria_midi",
]
