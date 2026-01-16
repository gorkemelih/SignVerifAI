#!/usr/bin/env python3
"""
Clean problematic images from dataset.

Removes: blank/empty images, low-contrast images
Keeps: duplicates (same person variations are useful)
"""

import os
import shutil
from pathlib import Path
import pandas as pd

# Paths
DATA_PATH = Path("data_processed")
QUALITY_DIR = Path("outputs/quality_analysis")
BACKUP_DIR = Path("data_removed_backup")


def clean_dataset():
    """Remove problematic images and update metadata."""
    print("=" * 60)
    print("CLEANING DATASET")
    print("=" * 60)
    
    # Load issue reports
    blank_path = QUALITY_DIR / "issues_blank.csv"
    
    if not blank_path.exists():
        print("ERROR: Run analyze_quality.py first!")
        return
    
    # Get blank images - fix path format
    blank_df = pd.read_csv(blank_path)
    # Remove "data_processed/" prefix to match metadata format
    blank_paths = blank_df["path"].str.replace("data_processed/", "", regex=False).tolist()
    print(f"Blank images to remove: {len(blank_paths)}")
    
    # Create paths to remove (only blank, not duplicates)
    paths_to_remove = set(blank_paths)
    
    print(f"\nTotal images to remove: {len(paths_to_remove)}")
    
    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Move problematic images to backup
    print("\n[1/3] Moving problematic images to backup...")
    moved = 0
    for rel_path in paths_to_remove:
        src_path = DATA_PATH / rel_path
        if src_path.exists():
            # Create backup path
            backup_path = BACKUP_DIR / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(src_path), str(backup_path))
            moved += 1
    
    print(f"   Moved {moved} files to {BACKUP_DIR}")
    
    # Update metadata
    print("\n[2/3] Updating metadata...")
    metadata_path = DATA_PATH / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    
    original_count = len(metadata)
    
    # Filter out removed paths
    metadata = metadata[~metadata["path"].isin(paths_to_remove)]
    
    # Save updated metadata
    metadata.to_csv(metadata_path, index=False)
    
    new_count = len(metadata)
    removed_count = original_count - new_count
    
    print(f"   Original: {original_count}")
    print(f"   Removed: {removed_count}")
    print(f"   Remaining: {new_count}")
    
    # Print summary by source
    print("\n[3/3] Final dataset summary...")
    print(f"\n📊 BY SOURCE:")
    for source in metadata["source"].unique():
        source_df = metadata[metadata["source"] == source]
        print(f"   {source}: {len(source_df)}")
    
    print(f"\n📊 BY LABEL:")
    print(metadata["label"].value_counts().to_string())
    
    print(f"\n📊 UNIQUE PERSONS: {metadata['person_id'].nunique()}")
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print(f"Removed {removed_count} problematic images")
    print(f"Clean dataset: {new_count} images")
    print("=" * 60)
    
    return metadata


if __name__ == "__main__":
    clean_dataset()
