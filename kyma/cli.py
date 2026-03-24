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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kyma")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_kinds = ("model", "eval", "training")

    list_parser = subparsers.add_parser("list-configs")
    list_parser.add_argument("kind", choices=config_kinds)

    print_parser = subparsers.add_parser("print-config")
    print_parser.add_argument("kind", choices=config_kinds)
    print_parser.add_argument("name")

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

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
