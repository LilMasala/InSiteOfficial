#!/usr/bin/env python3
"""Unified Chamelia training orchestrator entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.orchestrator import UnifiedTrainingOrchestrator, load_orchestrator_config


def _serialize_for_yaml(value: Any) -> Any:
    """Convert nested orchestrator outputs into YAML-friendly primitives."""
    if isinstance(value, dict):
        return {str(key): _serialize_for_yaml(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_yaml(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the unified Chamelia orchestrator")
    parser.add_argument("--config", required=True, help="Path to orchestrator YAML config")
    parser.add_argument("--device", default=None, help="Override config device")
    parser.add_argument("--run-dir", default=None, help="Override config run directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help="Restrict execution to one or more named domains",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="Print configured domains and exit",
    )
    return parser.parse_args()


def main() -> int:
    """Load config, run the orchestrator, and persist a summary report."""
    args = parse_args()
    config = load_orchestrator_config(args.config)
    if args.device is not None:
        config.device = args.device
    if args.run_dir is not None:
        config.run_dir = args.run_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.domain:
        requested = set(args.domain)
        config.domains = [domain for domain in config.domains if domain.name in requested]
    if args.list_domains:
        for domain in config.domains:
            print(domain.name)
        return 0
    if not config.domains:
        raise ValueError("No domains selected for orchestrator run.")

    orchestrator = UnifiedTrainingOrchestrator(config)
    results = orchestrator.run()
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "orchestrator_results.yaml"
    result_path.write_text(yaml.safe_dump(_serialize_for_yaml(results), sort_keys=False))
    print(f"Results written to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
