"""Dataset splitting module for SignVerify.

Creates person-disjoint train/val/test splits for signature verification.
"""

import random
from pathlib import Path
from typing import Optional

import pandas as pd

from signverify.config import PathConfig, SplitConfig
from signverify.utils.io import read_csv, write_csv
from signverify.utils.logging import get_logger
from signverify.utils.seed import set_seed

logger = get_logger(__name__)


def get_person_ids(df: pd.DataFrame) -> list[str]:
    """Get unique person IDs from metadata."""
    return sorted(df["person_id"].unique().tolist())


def split_persons(
    person_ids: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split person IDs into train/val/test sets.
    
    Args:
        person_ids: List of person IDs
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    set_seed(seed)
    
    ids = person_ids.copy()
    random.shuffle(ids)
    
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    
    return train_ids, val_ids, test_ids


def create_split_dataframes(
    df: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create DataFrames for each split based on person IDs.
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = df[df["person_id"].isin(train_ids)].copy()
    val_df = df[df["person_id"].isin(val_ids)].copy()
    test_df = df[df["person_id"].isin(test_ids)].copy()
    
    return train_df, val_df, test_df


def log_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> None:
    """Log split statistics."""
    logger.info("=" * 50)
    logger.info("SPLIT SUMMARY")
    logger.info("=" * 50)
    
    for name, df, ids in [
        ("Train", train_df, train_ids),
        ("Val", val_df, val_ids),
        ("Test", test_df, test_ids),
    ]:
        n_genuine = len(df[df["label"] == "genuine"])
        n_forged = len(df[df["label"] == "forged"])
        logger.info(
            f"{name:5s}: {len(ids):3d} persons | "
            f"{len(df):5d} images | "
            f"genuine={n_genuine:4d} | forged={n_forged:4d}"
        )
    
    logger.info("=" * 50)


def run_split(
    paths: Optional[PathConfig] = None,
    config: Optional[SplitConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run person-disjoint split pipeline.
    
    Args:
        paths: Path configuration
        config: Split configuration
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if paths is None:
        paths = PathConfig()
    if config is None:
        config = SplitConfig()
    
    logger.info("Starting person-disjoint split...")
    logger.info(f"Ratios: train={config.train_ratio}, val={config.val_ratio}, test={config.test_ratio}")
    
    # Read metadata
    df = read_csv(paths.metadata_csv)
    logger.info(f"Loaded {len(df)} images from metadata")
    
    # Get person IDs and split
    person_ids = get_person_ids(df)
    logger.info(f"Found {len(person_ids)} unique persons")
    
    train_ids, val_ids, test_ids = split_persons(
        person_ids,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    
    # Create split DataFrames
    train_df, val_df, test_df = create_split_dataframes(df, train_ids, val_ids, test_ids)
    
    # Log summary
    log_split_summary(train_df, val_df, test_df, train_ids, val_ids, test_ids)
    
    # Save splits
    paths.splits.mkdir(parents=True, exist_ok=True)
    
    write_csv(train_df, paths.splits / "train.csv")
    write_csv(val_df, paths.splits / "val.csv")
    write_csv(test_df, paths.splits / "test.csv")
    
    logger.info(f"Splits saved to: {paths.splits}")
    
    return train_df, val_df, test_df
