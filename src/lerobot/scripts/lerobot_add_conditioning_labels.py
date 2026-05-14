#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add episode-level conditioning labels to a cached LeRobot dataset.

This script edits the episode metadata parquet files in-place so that
``LeRobotDataset.__getitem__`` can return ``conditioning`` alongside
``task`` / ``task_index``.

Use this when you already recorded a dataset and want to retrofit conditioning
without re-recording.

Examples:

Assign a repeating episode-order pattern:
    python -m lerobot.scripts.lerobot_add_conditioning_labels \
        --repo-id jogarulfo/dataset_MVP_store_cardboard \
        --conditioning-pattern "[1, 2, 1, 1]"

Assign exact labels for a few episodes:
    python -m lerobot.scripts.lerobot_add_conditioning_labels \
        --repo-id jogarulfo/dataset_MVP_store_cardboard \
        --episode-conditioning '{"0": 1, "1": 2, "2": 1, "3": 1}'

If your dataset is stored in a custom local cache folder, pass ``--root``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.utils.utils import init_logging


def _parse_int_list(value: str) -> list[int]:
    value = value.strip()
    if not value:
        raise ValueError("conditioning pattern cannot be empty")
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_episode_mapping(value: str) -> dict[int, int]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("episode conditioning must be a JSON object mapping episode_index to conditioning")
    return {int(key): int(val) for key, val in parsed.items()}


def _build_pattern_mapping(total_episodes: int, pattern: list[int]) -> dict[int, int]:
    if not pattern:
        raise ValueError("conditioning pattern cannot be empty")
    return {episode_index: int(pattern[episode_index % len(pattern)]) for episode_index in range(total_episodes)}


def _apply_conditioning_to_file(
    parquet_path: Path,
    conditioning_by_episode: dict[int, int],
    overwrite_existing: bool,
    dry_run: bool,
) -> tuple[int, int]:
    df = pd.read_parquet(parquet_path)
    if "episode_index" not in df.columns:
        raise ValueError(f"{parquet_path} does not contain an episode_index column")

    if "conditioning" in df.columns and not overwrite_existing and df["conditioning"].notna().any():
        raise ValueError(
            f"{parquet_path} already contains conditioning values. Pass --overwrite-existing to replace them."
        )

    mapped = df["episode_index"].map(conditioning_by_episode)
    if mapped.isna().any():
        missing = sorted(set(int(ep) for ep in df.loc[mapped.isna(), "episode_index"].unique()))
        raise ValueError(f"Missing conditioning labels for episodes: {missing}")

    df["conditioning"] = mapped.astype(int)
    if not dry_run:
        df.to_parquet(parquet_path, index=False)
    return len(df), df["episode_index"].nunique()


def add_conditioning_labels(
    repo_id: str,
    root: str | Path | None = None,
    episode_conditioning: dict[int, int] | None = None,
    conditioning_pattern: list[int] | None = None,
    overwrite_existing: bool = False,
    dry_run: bool = False,
) -> None:
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    dataset_root = meta.root
    if meta.episodes is None:
        meta.ensure_readable()

    if episode_conditioning is None and conditioning_pattern is None:
        raise ValueError("Provide either episode_conditioning or conditioning_pattern")

    if episode_conditioning is None:
        episode_conditioning = _build_pattern_mapping(meta.total_episodes, conditioning_pattern or [])

    episodes_dir = dataset_root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {episodes_dir}")

    logging.info("Applying conditioning labels to %s", dataset_root)
    logging.info("Episodes covered: %s", sorted(episode_conditioning.keys())[:10])

    total_rows = 0
    total_files = 0
    for parquet_path in parquet_files:
        rows, unique_episodes = _apply_conditioning_to_file(
            parquet_path=parquet_path,
            conditioning_by_episode=episode_conditioning,
            overwrite_existing=overwrite_existing,
            dry_run=dry_run,
        )
        total_rows += rows
        total_files += 1
        logging.info(
            "%s: updated %s rows across %s episode(s)%s",
            parquet_path.relative_to(dataset_root),
            rows,
            unique_episodes,
            " (dry run)" if dry_run else "",
        )

    logging.info("Done: %s file(s), %s row(s)%s", total_files, total_rows, " (dry run)" if dry_run else "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Dataset repository id, e.g. user/dataset")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset root. Defaults to the standard LeRobot cache location.",
    )
    parser.add_argument(
        "--episode-conditioning",
        type=str,
        default=None,
        help='Exact episode->conditioning mapping as JSON, e.g. "{\"0\": 1, \"1\": 2}"',
    )
    parser.add_argument(
        "--conditioning-pattern",
        type=str,
        default=None,
        help='Repeating conditioning pattern in episode order, e.g. "[1, 2, 1, 1]" or "1,2,1,1"',
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing conditioning column instead of failing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the mapping without writing files.",
    )
    return parser


def main() -> None:
    init_logging()
    parser = build_parser()
    args = parser.parse_args()

    episode_conditioning = None
    conditioning_pattern = None
    if args.episode_conditioning is not None:
        episode_conditioning = _parse_episode_mapping(args.episode_conditioning)
    if args.conditioning_pattern is not None:
        conditioning_pattern = _parse_int_list(args.conditioning_pattern)

    add_conditioning_labels(
        repo_id=args.repo_id,
        root=args.root,
        episode_conditioning=episode_conditioning,
        conditioning_pattern=conditioning_pattern,
        overwrite_existing=args.overwrite_existing,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()