from __future__ import annotations

import json
from email.message import Message
from pathlib import Path
from urllib import error

import pytest

from kyma.data import (
    ARIA_MIDI_DEFAULT_ROOT,
    build_aria_midi_download_plan,
    download_aria_midi,
)


def test_build_aria_midi_download_plan_defaults_to_pruned_subset() -> None:
    plan = build_aria_midi_download_plan()

    assert plan.subset == "pruned"
    assert plan.root.endswith("artifacts/data/aria-midi/pruned")
    assert plan.archive_path.endswith("aria-midi-v1-pruned-ext.tar.gz")
    assert plan.archive_url.endswith("aria-midi-v1-pruned-ext.tar.gz?download=true")
    assert plan.preprocess_path is not None
    assert plan.preprocess_path.endswith("light-preprocess.json")
    assert plan.license_name == "CC-BY-NC-SA 4.0"


def test_build_aria_midi_download_plan_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="Unknown subset"):
        build_aria_midi_download_plan(subset="bad")


def test_download_aria_midi_requires_explicit_license_acceptance() -> None:
    with pytest.raises(ValueError, match="license acceptance"):
        download_aria_midi(dry_run=False)


def test_download_aria_midi_dry_run_returns_plan() -> None:
    manifest = download_aria_midi(
        dry_run=True,
        root=ARIA_MIDI_DEFAULT_ROOT,
    )

    assert manifest["subset"] == "pruned"
    assert manifest["archive_url"].endswith("?download=true")


def test_download_aria_midi_writes_manifest_and_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloads: list[tuple[str, Path]] = []

    def fake_download(
        url: str,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        del overwrite
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(f"downloaded from {url}\n", encoding="utf-8")
        downloads.append((url, destination_path))

    monkeypatch.setattr("kyma.data.aria_midi._download_to_path", fake_download)

    manifest = download_aria_midi(
        subset="pruned",
        root=tmp_path,
        accept_license=True,
    )

    root = tmp_path / "pruned"
    manifest_path = root / "manifest.json"
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["subset"] == "pruned"
    assert payload["paths"]["archive"].endswith("aria-midi-v1-pruned-ext.tar.gz")
    assert len(downloads) == 4
    assert manifest["paths"]["root"] == str(root)


def test_download_to_path_treats_http_416_as_already_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "existing.bin"
    destination.write_bytes(b"complete")

    def fake_urlopen(_request: object) -> object:
        headers = Message()
        raise error.HTTPError(
            url="https://example.com/file",
            code=416,
            msg="Range Not Satisfiable",
            hdrs=headers,
            fp=None,
        )

    monkeypatch.setattr("kyma.data.aria_midi.request.urlopen", fake_urlopen)

    plan = build_aria_midi_download_plan(root=tmp_path)
    for path_str in (
        plan.archive_path,
        plan.disclaimer_path,
        plan.readme_path,
        plan.preprocess_path,
    ):
        assert path_str is not None
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")

    manifest = download_aria_midi(
        root=tmp_path,
        accept_license=True,
    )
    assert Path(manifest["paths"]["archive"]).read_bytes() == b"complete"
