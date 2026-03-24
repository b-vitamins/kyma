"""Download and local-cache helpers for the public Aria-MIDI dataset."""

from __future__ import annotations

import json
import random
import shutil
import sys
import tarfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib import error, request

from tqdm import tqdm

from kyma.data.piece_cache import save_piece_cache
from kyma.data.pieces import tokenize_midi_record
from kyma.data.tokenization import get_abs_tokenizer

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


def _subset_spec(subset: str) -> AriaMidiSubsetSpec:
    if subset not in ARIA_MIDI_SUBSETS:
        choices = ", ".join(sorted(ARIA_MIDI_SUBSETS))
        raise ValueError(f"Unknown subset {subset!r}. Expected one of: {choices}.")
    return ARIA_MIDI_SUBSETS[subset]


def _archive_dirname(spec: AriaMidiSubsetSpec) -> str:
    return spec.archive_filename.removesuffix(".tar.gz")


def _load_mididict_class() -> type[Any]:
    try:
        midi_module = import_module("ariautils.midi")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ariautils is required for building Aria-MIDI piece caches."
        ) from exc
    return midi_module.MidiDict


def resolve_aria_midi_subset_root(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
) -> Path:
    _subset_spec(subset)
    return Path(root) / subset


def resolve_aria_midi_archive_path(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
) -> Path:
    spec = _subset_spec(subset)
    return (
        resolve_aria_midi_subset_root(subset=subset, root=root) / spec.archive_filename
    )


def resolve_aria_midi_extract_root(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
) -> Path:
    _subset_spec(subset)
    return resolve_aria_midi_subset_root(subset=subset, root=root) / "extracted"


def resolve_aria_midi_dataset_root(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
    extracted_root: str | Path | None = None,
) -> Path:
    spec = _subset_spec(subset)
    base_root = (
        resolve_aria_midi_extract_root(subset=subset, root=root)
        if extracted_root is None
        else Path(extracted_root)
    )
    return base_root / _archive_dirname(spec)


def resolve_aria_midi_piece_cache_path(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
) -> Path:
    _subset_spec(subset)
    return resolve_aria_midi_subset_root(subset=subset, root=root) / "piece-cache.jsonl"


def build_aria_midi_download_plan(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
) -> AriaMidiDownloadPlan:
    """Build a local-cache plan for one published Aria-MIDI subset."""

    spec = _subset_spec(subset)
    root_path = resolve_aria_midi_subset_root(subset=subset, root=root)
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


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        member_path.relative_to(destination_resolved)
    archive.extractall(destination)


def extract_aria_midi_archive(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract a downloaded Aria-MIDI archive into the local cache tree."""

    spec = _subset_spec(subset)
    archive_path = resolve_aria_midi_archive_path(subset=subset, root=root)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Aria-MIDI archive not found: {archive_path}")

    extract_root = (
        resolve_aria_midi_extract_root(subset=subset, root=root)
        if output_dir is None
        else Path(output_dir)
    )
    dataset_root = extract_root / _archive_dirname(spec)
    manifest_path = (
        resolve_aria_midi_subset_root(
            subset=subset,
            root=root,
        )
        / "extract-manifest.json"
    )

    if dataset_root.exists():
        if overwrite:
            shutil.rmtree(dataset_root)
        else:
            manifest = {
                "dataset": "aria-midi",
                "subset": subset,
                "archive_path": str(archive_path),
                "extract_root": str(extract_root),
                "dataset_root": str(dataset_root),
                "extracted_at_utc": datetime.now(UTC).isoformat(),
                "overwrite": False,
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return manifest

    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        _safe_extract_tar(archive, extract_root)

    manifest = {
        "dataset": "aria-midi",
        "subset": subset,
        "archive_path": str(archive_path),
        "extract_root": str(extract_root),
        "dataset_root": str(dataset_root),
        "extracted_at_utc": datetime.now(UTC).isoformat(),
        "overwrite": overwrite,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_aria_midi_piece_cache(
    *,
    subset: str = "pruned",
    root: str | Path = ARIA_MIDI_DEFAULT_ROOT,
    extracted_root: str | Path | None = None,
    output_path: str | Path | None = None,
    tokenizer_config_path: str | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    random_seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a JSONL piece cache from extracted Aria-MIDI files."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    dataset_root = resolve_aria_midi_dataset_root(
        subset=subset,
        root=root,
        extracted_root=extracted_root,
    )
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Expected extracted Aria-MIDI data directory at {data_root}"
        )

    cache_path = (
        resolve_aria_midi_piece_cache_path(subset=subset, root=root)
        if output_path is None
        else Path(output_path)
    )
    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"Piece cache already exists: {cache_path}")

    midi_paths = sorted(data_root.rglob("*.mid"))
    if shuffle:
        random.Random(random_seed).shuffle(midi_paths)
    if limit is not None:
        midi_paths = midi_paths[:limit]

    midi_dict_cls = _load_mididict_class()
    tokenizer = get_abs_tokenizer(config_path=tokenizer_config_path)
    stats = {"selected": 0, "failed": 0}
    error_samples: list[dict[str, str]] = []

    def _iter_pieces():
        for midi_path in midi_paths:
            stats["selected"] += 1
            try:
                midi_dict = midi_dict_cls.from_midi(str(midi_path))
                relative_path = str(midi_path.relative_to(dataset_root))
                yield tokenize_midi_record(
                    midi_dict,
                    tokenizer=tokenizer,
                    piece_id=relative_path,
                    metadata={"subset": subset},
                    source_path=str(midi_path),
                )
            except Exception as exc:
                stats["failed"] += 1
                if len(error_samples) < 16:
                    error_samples.append(
                        {
                            "path": str(midi_path),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    written = save_piece_cache(_iter_pieces(), cache_path)
    manifest = {
        "dataset": "aria-midi",
        "subset": subset,
        "dataset_root": str(dataset_root),
        "cache_path": str(cache_path),
        "built_at_utc": datetime.now(UTC).isoformat(),
        "selected_files": stats["selected"],
        "pieces_written": written,
        "failed_files": stats["failed"],
        "shuffle": shuffle,
        "random_seed": random_seed,
        "limit": limit,
        "tokenizer_config_path": tokenizer_config_path,
        "error_samples": error_samples,
    }
    cache_manifest_path = Path(f"{cache_path}.manifest.json")
    cache_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


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
    "build_aria_midi_piece_cache",
    "download_aria_midi",
    "extract_aria_midi_archive",
    "resolve_aria_midi_archive_path",
    "resolve_aria_midi_dataset_root",
    "resolve_aria_midi_extract_root",
    "resolve_aria_midi_piece_cache_path",
    "resolve_aria_midi_subset_root",
]
