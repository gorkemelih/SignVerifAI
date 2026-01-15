"""Siamese pair generation module for SignVerify.

Generates positive and negative pairs for contrastive learning.
Supports with-replacement sampling to reach target pair counts.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from signverify.config import PairConfig, PathConfig
from signverify.utils.io import read_csv, write_csv
from signverify.utils.logging import get_logger
from signverify.utils.seed import set_seed

logger = get_logger(__name__)


def group_by_person(df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """
    Group images by person and label.
    
    Returns:
        Dict[person_id, {"genuine": [paths], "forged": [paths]}]
    """
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"genuine": [], "forged": []})
    
    for _, row in df.iterrows():
        groups[row["person_id"]][row["label"]].append(row["path"])
    
    return dict(groups)


def sample_positive_pairs(
    groups: dict[str, dict[str, list[str]]],
    n_pairs: int,
) -> list[tuple[str, str, int, str]]:
    """
    Sample positive pairs with replacement: genuine-genuine same person.
    
    Args:
        groups: Grouped images by person
        n_pairs: Target number of pairs
    
    Returns:
        List of (img1_path, img2_path, target=1, pair_type)
    """
    pairs = []
    
    # Build pool of valid persons (at least 2 genuines)
    valid_persons = [
        (pid, labels["genuine"]) 
        for pid, labels in groups.items() 
        if len(labels["genuine"]) >= 2
    ]
    
    if not valid_persons:
        logger.warning("No valid persons for positive pairs")
        return pairs
    
    for _ in range(n_pairs):
        # Sample random person
        pid, genuines = random.choice(valid_persons)
        # Sample two different genuines
        img1, img2 = random.sample(genuines, 2)
        pairs.append((img1, img2, 1, "genuine_genuine_same"))
    
    return pairs


def sample_negative_same_person(
    groups: dict[str, dict[str, list[str]]],
    n_pairs: int,
) -> list[tuple[str, str, int, str]]:
    """
    Sample negative pairs with replacement: genuine-forged same person.
    
    Args:
        groups: Grouped images by person
        n_pairs: Target number of pairs
    
    Returns:
        List of (img1_path, img2_path, target=0, pair_type)
    """
    pairs = []
    
    # Build pool of valid persons
    valid_persons = [
        (pid, labels["genuine"], labels["forged"]) 
        for pid, labels in groups.items() 
        if labels["genuine"] and labels["forged"]
    ]
    
    if not valid_persons:
        logger.warning("No valid persons for genuine-forged pairs")
        return pairs
    
    for _ in range(n_pairs):
        pid, genuines, forgeds = random.choice(valid_persons)
        img1 = random.choice(genuines)
        img2 = random.choice(forgeds)
        pairs.append((img1, img2, 0, "genuine_forged_same"))
    
    return pairs


def sample_negative_diff_person(
    groups: dict[str, dict[str, list[str]]],
    n_pairs: int,
) -> list[tuple[str, str, int, str]]:
    """
    Sample negative pairs with replacement: genuine-genuine different persons.
    
    Args:
        groups: Grouped images by person
        n_pairs: Target number of pairs
    
    Returns:
        List of (img1_path, img2_path, target=0, pair_type)
    """
    pairs = []
    
    # Build pool of all persons with genuines
    valid_persons = [
        (pid, labels["genuine"]) 
        for pid, labels in groups.items() 
        if labels["genuine"]
    ]
    
    if len(valid_persons) < 2:
        logger.warning("Not enough persons for different-person pairs")
        return pairs
    
    for _ in range(n_pairs):
        # Sample two different persons
        (pid1, genuines1), (pid2, genuines2) = random.sample(valid_persons, 2)
        img1 = random.choice(genuines1)
        img2 = random.choice(genuines2)
        pairs.append((img1, img2, 0, "genuine_genuine_diff"))
    
    return pairs


def generate_balanced_pairs(
    df: pd.DataFrame,
    target_pairs: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate balanced pairs with replacement to reach target count.
    
    Ratio: Positive : Negative = 1 : 2
    Negative: 50% genuine-forged, 50% genuine-genuine-diff
    
    Args:
        df: Split DataFrame
        target_pairs: Target total number of pairs
        seed: Random seed
    
    Returns:
        DataFrame with columns: img1_path, img2_path, target, pair_type
    """
    set_seed(seed)
    
    groups = group_by_person(df)
    
    # Calculate counts: 1:2 ratio (pos:neg)
    n_positive = target_pairs // 3
    n_negative = target_pairs - n_positive
    n_neg_same = n_negative // 2
    n_neg_diff = n_negative - n_neg_same
    
    logger.info(f"  Target distribution: {n_positive} pos + {n_neg_same} neg-same + {n_neg_diff} neg-diff")
    
    # Sample pairs with replacement
    positive = sample_positive_pairs(groups, n_positive)
    negative_same = sample_negative_same_person(groups, n_neg_same)
    negative_diff = sample_negative_diff_person(groups, n_neg_diff)
    
    all_pairs = positive + negative_same + negative_diff
    random.shuffle(all_pairs)
    
    # Create DataFrame
    pairs_df = pd.DataFrame(
        all_pairs,
        columns=["img1_path", "img2_path", "target", "pair_type"],
    )
    
    return pairs_df


def log_pair_summary(pairs_df: pd.DataFrame, split_name: str) -> None:
    """Log pair generation statistics."""
    logger.info(f"\n{split_name} Pairs Summary:")
    logger.info(f"  Total pairs: {len(pairs_df)}")
    logger.info(f"  Positive (target=1): {len(pairs_df[pairs_df['target'] == 1])}")
    logger.info(f"  Negative (target=0): {len(pairs_df[pairs_df['target'] == 0])}")
    logger.info("  By type:")
    for pair_type, count in pairs_df["pair_type"].value_counts().items():
        logger.info(f"    {pair_type}: {count}")


def run_pairs(
    paths: Optional[PathConfig] = None,
    config: Optional[PairConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate pairs for train and val splits.
    
    Args:
        paths: Path configuration
        config: Pair configuration
    
    Returns:
        Tuple of (train_pairs_df, val_pairs_df)
    """
    if paths is None:
        paths = PathConfig()
    if config is None:
        config = PairConfig()
    
    logger.info("Starting pair generation (with replacement)...")
    
    # Check splits exist
    train_path = paths.splits / "train.csv"
    val_path = paths.splits / "val.csv"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Splits not found. Run 'signverify split' first.")
    
    # Load splits
    train_df = read_csv(train_path)
    val_df = read_csv(val_path)
    
    logger.info(f"Loaded train: {len(train_df)} images, val: {len(val_df)} images")
    logger.info(f"Target pairs: train={config.max_train_pairs}, val={config.max_val_pairs}")
    logger.info("Ratio: 1:2 (positive:negative), negative split 50/50")
    
    # Generate pairs with replacement
    logger.info("\nGenerating train pairs...")
    train_pairs = generate_balanced_pairs(train_df, config.max_train_pairs, config.seed)
    
    logger.info("\nGenerating val pairs...")
    val_pairs = generate_balanced_pairs(val_df, config.max_val_pairs, config.seed + 1)
    
    log_pair_summary(train_pairs, "Train")
    log_pair_summary(val_pairs, "Val")
    
    # Save pairs
    paths.pairs.mkdir(parents=True, exist_ok=True)
    
    write_csv(train_pairs, paths.pairs / "pairs_train.csv")
    write_csv(val_pairs, paths.pairs / "pairs_val.csv")
    
    logger.info(f"\nPairs saved to: {paths.pairs}")
    
    return train_pairs, val_pairs
