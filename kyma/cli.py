"""Repository utility CLI for Kyma."""

from __future__ import annotations

import argparse
import json

from kyma.config import (
    list_eval_configs,
    list_model_configs,
    list_training_configs,
    load_eval_config,
    load_model_config,
    load_training_config,
)
from kyma.data import (
    build_aria_midi_piece_cache,
    download_aria_midi,
    extract_aria_midi_archive,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kyma")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_kinds = ("model", "eval", "training")

    list_parser = subparsers.add_parser("list-configs")
    list_parser.add_argument("kind", choices=config_kinds)

    print_parser = subparsers.add_parser("print-config")
    print_parser.add_argument("kind", choices=config_kinds)
    print_parser.add_argument("name")

    download_parser = subparsers.add_parser("download-aria-midi")
    download_parser.add_argument(
        "--subset",
        choices=("full", "pruned", "deduped", "unique"),
        default="pruned",
    )
    download_parser.add_argument(
        "--root",
        default="artifacts/data/aria-midi",
    )
    download_parser.add_argument(
        "--accept-license",
        action="store_true",
        help="acknowledge the CC-BY-NC-SA 4.0 license and upstream disclaimer",
    )
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    download_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved download plan without fetching files",
    )

    extract_parser = subparsers.add_parser("extract-aria-midi")
    extract_parser.add_argument(
        "--subset",
        choices=("full", "pruned", "deduped", "unique"),
        default="pruned",
    )
    extract_parser.add_argument(
        "--root",
        default="artifacts/data/aria-midi",
    )
    extract_parser.add_argument(
        "--output-dir",
        default=None,
    )
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    cache_parser = subparsers.add_parser("build-aria-midi-piece-cache")
    cache_parser.add_argument(
        "--subset",
        choices=("full", "pruned", "deduped", "unique"),
        default="pruned",
    )
    cache_parser.add_argument(
        "--root",
        default="artifacts/data/aria-midi",
    )
    cache_parser.add_argument(
        "--extracted-root",
        default=None,
    )
    cache_parser.add_argument(
        "--output-path",
        default=None,
    )
    cache_parser.add_argument(
        "--tokenizer-config-path",
        default=None,
    )
    cache_parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    cache_parser.add_argument(
        "--shuffle",
        action="store_true",
    )
    cache_parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
    )
    cache_parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "list-configs":
        if args.kind == "model":
            configs = list_model_configs()
        elif args.kind == "eval":
            configs = list_eval_configs()
        else:
            configs = list_training_configs()
        print("\n".join(configs))
        return

    if args.command == "print-config":
        if args.kind == "model":
            config = load_model_config(args.name)
        elif args.kind == "eval":
            config = load_eval_config(args.name)
        else:
            config = load_training_config(args.name)
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    if args.command == "download-aria-midi":
        manifest = download_aria_midi(
            subset=args.subset,
            root=args.root,
            accept_license=args.accept_license,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "extract-aria-midi":
        manifest = extract_aria_midi_archive(
            subset=args.subset,
            root=args.root,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "build-aria-midi-piece-cache":
        manifest = build_aria_midi_piece_cache(
            subset=args.subset,
            root=args.root,
            extracted_root=args.extracted_root,
            output_path=args.output_path,
            tokenizer_config_path=args.tokenizer_config_path,
            limit=args.limit,
            shuffle=args.shuffle,
            random_seed=args.random_seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
