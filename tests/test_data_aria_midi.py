from __future__ import annotations

import json
import tarfile
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib import error

import pytest
import torch

from kyma.data import (
    ARIA_MIDI_DEFAULT_ROOT,
    KymaTimeFeatures,
    KymaTokenizedPiece,
    build_aria_midi_download_plan,
    build_aria_midi_piece_cache,
    download_aria_midi,
    extract_aria_midi_archive,
    load_piece_cache,
)
from kyma.data.aria_midi import RemoteFileMetadata


def _fake_probe_remote_file(_url: str) -> RemoteFileMetadata:
    return RemoteFileMetadata(size_bytes=1234, etag='"etag-value"')


def _fake_probe_complete_file(_url: str) -> RemoteFileMetadata:
    return RemoteFileMetadata(size_bytes=len(b"complete"), etag=None)


def _fake_probe_short_file(_url: str) -> RemoteFileMetadata:
    return RemoteFileMetadata(size_bytes=16, etag=None)


def _fake_probe_manifestless_extract(_url: str) -> RemoteFileMetadata:
    return RemoteFileMetadata(size_bytes=999, etag=None)


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
        expected_size_bytes: int | None = None,
    ) -> None:
        del overwrite, expected_size_bytes
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(f"downloaded from {url}\n", encoding="utf-8")
        downloads.append((url, destination_path))

    monkeypatch.setattr("kyma.data.aria_midi._download_to_path", fake_download)
    monkeypatch.setattr(
        "kyma.data.aria_midi._probe_remote_file",
        _fake_probe_remote_file,
    )

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
    assert payload["archive_expected_size_bytes"] == 1234
    assert payload["archive_etag"] == '"etag-value"'
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
    monkeypatch.setattr(
        "kyma.data.aria_midi._probe_remote_file",
        _fake_probe_complete_file,
    )

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


def test_download_aria_midi_rejects_short_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_urlopen(_request: object) -> object:
        class FakeResponse:
            status = 200
            headers = Message()

            def __enter__(self) -> FakeResponse:
                self.headers["Content-Length"] = "4"
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _size: int) -> bytes:
                if hasattr(self, "_done"):
                    return b""
                self._done = True
                return b"tiny"

        return FakeResponse()

    monkeypatch.setattr("kyma.data.aria_midi.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "kyma.data.aria_midi._probe_remote_file",
        _fake_probe_short_file,
    )

    with pytest.raises(ValueError, match="does not match the expected upstream size"):
        download_aria_midi(
            root=tmp_path,
            accept_license=True,
        )


def test_extract_aria_midi_archive_unpacks_expected_tree(tmp_path: Path) -> None:
    archive_path = tmp_path / "pruned" / "aria-midi-v1-pruned-ext.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, payload in (
            ("aria-midi-v1-pruned-ext/data/example.mid", b"midi"),
            ("aria-midi-v1-pruned-ext/metadata.json", b"{}"),
        ):
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))

    manifest = extract_aria_midi_archive(root=tmp_path)

    dataset_root = tmp_path / "pruned" / "extracted" / "aria-midi-v1-pruned-ext"
    assert manifest["dataset_root"] == str(dataset_root)
    assert (dataset_root / "data" / "example.mid").is_file()
    assert (tmp_path / "pruned" / "extract-manifest.json").is_file()


def test_extract_aria_midi_archive_rejects_incomplete_local_archive(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "pruned"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / "aria-midi-v1-pruned-ext.tar.gz"
    archive_path.write_bytes(b"short")
    (archive_root / "manifest.json").write_text(
        json.dumps({"archive_expected_size_bytes": 999}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete or stale"):
        extract_aria_midi_archive(root=tmp_path)


def test_extract_aria_midi_archive_uses_remote_probe_when_manifest_lacks_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "pruned"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / "aria-midi-v1-pruned-ext.tar.gz"
    archive_path.write_bytes(b"short")
    (archive_root / "manifest.json").write_text(
        json.dumps({"subset": "pruned"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kyma.data.aria_midi._probe_remote_file",
        _fake_probe_manifestless_extract,
    )

    with pytest.raises(ValueError, match="incomplete or stale"):
        extract_aria_midi_archive(root=tmp_path)


def test_build_aria_midi_piece_cache_serializes_tokenized_pieces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_root = (
        tmp_path / "pruned" / "extracted" / "aria-midi-v1-pruned-ext" / "data"
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "good.mid").write_bytes(b"midi")
    (dataset_root / "bad.mid").write_bytes(b"midi")

    class FakeMidiDict:
        @classmethod
        def from_midi(cls, path: str) -> dict[str, str]:
            return {"path": path}

    def fake_get_abs_tokenizer(*, config_path: str | None = None) -> object:
        del config_path
        return object()

    def fake_tokenize_midi_record(
        midi_dict: dict[str, str],
        *,
        tokenizer: object,
        piece_id: str | None = None,
        metadata: dict[str, str] | None = None,
        source_path: str | None = None,
        tokenize_kwargs: dict[str, object] | None = None,
    ) -> KymaTokenizedPiece:
        del tokenizer, metadata, tokenize_kwargs
        assert piece_id is not None
        assert source_path is not None
        if source_path.endswith("bad.mid"):
            raise ValueError("bad midi")
        return KymaTokenizedPiece(
            piece_id=piece_id,
            tokens=("<S>",),
            token_ids=torch.tensor([1], dtype=torch.long),
            time_features=KymaTimeFeatures(
                values=torch.zeros((1, 4), dtype=torch.float32),
                valid=torch.ones((1, 4), dtype=torch.bool),
            ),
            metadata={"path": midi_dict["path"]},
            source_path=source_path,
        )

    monkeypatch.setattr(
        "kyma.data.aria_midi._load_mididict_class",
        lambda: FakeMidiDict,
    )
    monkeypatch.setattr("kyma.data.aria_midi.get_abs_tokenizer", fake_get_abs_tokenizer)
    monkeypatch.setattr(
        "kyma.data.aria_midi.tokenize_midi_record",
        fake_tokenize_midi_record,
    )

    manifest = build_aria_midi_piece_cache(
        root=tmp_path,
        overwrite=True,
    )

    cache_path = tmp_path / "pruned" / "piece-cache.jsonl"
    pieces = load_piece_cache(cache_path)

    assert manifest["selected_files"] == 2
    assert manifest["pieces_written"] == 1
    assert manifest["failed_files"] == 1
    assert len(manifest["error_samples"]) == 1
    assert [piece.piece_id for piece in pieces] == ["data/good.mid"]
