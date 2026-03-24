#!/usr/bin/env python3
"""Branch-local runner for the RTX 3060 Kyma pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kyma.pilot import (
    DEFAULT_3060_CACHE_PATH,
    DEFAULT_3060_MODEL_CONFIG_PATH,
    DEFAULT_3060_OUTPUT_DIR,
    DEFAULT_3060_TRAINING_CONFIG_PATH,
    build_3060_pilot_cache,
    prepare_3060_pilot_run,
    train_3060_pilot,
    write_3060_pilot_summary,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtx3060-pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cache = subparsers.add_parser("build-cache")
    build_cache.add_argument("--subset", default="pruned")
    build_cache.add_argument("--root", default="artifacts/data/aria-midi")
    build_cache.add_argument("--extracted-root", default=None)
    build_cache.add_argument("--output-path", default=str(DEFAULT_3060_CACHE_PATH))
    build_cache.add_argument("--tokenizer-config-path", default=None)
    build_cache.add_argument("--max-pieces", type=int, default=16_000)
    build_cache.add_argument("--random-seed", type=int, default=0)
    build_cache.add_argument("--overwrite", action="store_true")

    plan = subparsers.add_parser("plan")
    plan.add_argument("--cache-path", default=str(DEFAULT_3060_CACHE_PATH))
    plan.add_argument(
        "--model-config-path",
        default=str(DEFAULT_3060_MODEL_CONFIG_PATH),
    )
    plan.add_argument(
        "--training-config-path",
        default=str(DEFAULT_3060_TRAINING_CONFIG_PATH),
    )
    plan.add_argument("--output-dir", default=str(DEFAULT_3060_OUTPUT_DIR))
    plan.add_argument("--tokenizer-config-path", default=None)
    plan.add_argument("--max-pieces", type=int, default=None)
    plan.add_argument("--val-ratio", type=float, default=0.02)
    plan.add_argument("--write-summary", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--cache-path", default=str(DEFAULT_3060_CACHE_PATH))
    train.add_argument(
        "--model-config-path",
        default=str(DEFAULT_3060_MODEL_CONFIG_PATH),
    )
    train.add_argument(
        "--training-config-path",
        default=str(DEFAULT_3060_TRAINING_CONFIG_PATH),
    )
    train.add_argument("--output-dir", default=str(DEFAULT_3060_OUTPUT_DIR))
    train.add_argument("--tokenizer-config-path", default=None)
    train.add_argument("--max-pieces", type=int, default=None)
    train.add_argument("--val-ratio", type=float, default=0.02)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "build-cache":
        manifest = build_3060_pilot_cache(
            subset=args.subset,
            root=args.root,
            extracted_root=args.extracted_root,
            output_path=args.output_path,
            tokenizer_config_path=args.tokenizer_config_path,
            max_pieces=args.max_pieces,
            random_seed=args.random_seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    prepared = prepare_3060_pilot_run(
        cache_path=args.cache_path,
        model_config_path=args.model_config_path,
        training_config_path=args.training_config_path,
        output_dir=args.output_dir,
        tokenizer_config_path=args.tokenizer_config_path,
        max_pieces=args.max_pieces,
        val_ratio=args.val_ratio,
    )

    if args.command == "plan":
        if args.write_summary:
            summary_path = write_3060_pilot_summary(
                prepared.summary,
                output_dir=Path(args.output_dir),
            )
            print(json.dumps({"summary_path": str(summary_path)}, indent=2))
        print(json.dumps(prepared.summary.to_dict(), indent=2, sort_keys=True))
        return

    if args.command == "train":
        report = train_3060_pilot(prepared)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
