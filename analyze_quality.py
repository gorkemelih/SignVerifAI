#!/usr/bin/env python3
"""
Data Quality Analysis for SignVerifAI.

Critical checks that can affect model generalization:
1. Empty/blank images (too much white)
2. Too small file sizes (corrupted)
3. Low contrast images
4. Duplicate/near-duplicate images
5. Unusual aspect ratios
6. Outlier detection
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Configuration
DATA_PATH = Path("data_processed")
METADATA_PATH = DATA_PATH / "metadata.csv"
OUTPUT_DIR = Path("outputs/quality_analysis")


def compute_image_hash(img_path: Path) -> str:
    """Compute perceptual hash of image for duplicate detection."""
    try:
        img = Image.open(img_path).convert("L")
        # Resize to 8x8 and compute average
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        arr = np.array(img)
        avg = arr.mean()
        # Create binary hash
        bits = (arr > avg).flatten()
        hash_str = ''.join(['1' if b else '0' for b in bits])
        return hash_str
    except Exception:
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hashes."""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def analyze_image(img_path: Path) -> dict:
    """Analyze a single image for quality issues."""
    result = {
        "path": str(img_path),
        "exists": False,
        "file_size": 0,
        "width": 0,
        "height": 0,
        "mean_pixel": 0,
        "std_pixel": 0,
        "min_pixel": 0,
        "max_pixel": 0,
        "white_ratio": 0,  # Ratio of white pixels
        "black_ratio": 0,  # Ratio of black pixels
        "signature_ratio": 0,  # Ratio of non-white pixels
        "contrast": 0,
        "is_blank": False,
        "is_low_contrast": False,
        "is_too_small": False,
        "is_corrupted": False,
        "phash": None,
    }
    
    if not img_path.exists():
        return result
    
    result["exists"] = True
    result["file_size"] = img_path.stat().st_size
    
    try:
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.float32)
        
        result["width"] = img.size[0]
        result["height"] = img.size[1]
        result["mean_pixel"] = arr.mean()
        result["std_pixel"] = arr.std()
        result["min_pixel"] = arr.min()
        result["max_pixel"] = arr.max()
        
        # White ratio (pixels > 250)
        white_threshold = 250
        result["white_ratio"] = (arr > white_threshold).sum() / arr.size
        
        # Black ratio (pixels < 10)
        black_threshold = 10
        result["black_ratio"] = (arr < black_threshold).sum() / arr.size
        
        # Signature ratio (non-white pixels that could be signature)
        result["signature_ratio"] = (arr < 200).sum() / arr.size
        
        # Contrast
        result["contrast"] = result["max_pixel"] - result["min_pixel"]
        
        # Quality flags
        result["is_blank"] = result["white_ratio"] > 0.98 or result["signature_ratio"] < 0.01
        result["is_low_contrast"] = result["std_pixel"] < 15 or result["contrast"] < 50
        result["is_too_small"] = result["file_size"] < 500
        
        # Perceptual hash
        result["phash"] = compute_image_hash(img_path)
        
    except Exception as e:
        result["is_corrupted"] = True
    
    return result


def find_duplicates(df: pd.DataFrame, threshold: int = 5) -> list:
    """Find near-duplicate images based on perceptual hash."""
    duplicates = []
    hashes = df[df["phash"].notna()][["path", "phash"]].values.tolist()
    
    n = len(hashes)
    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming_distance(hashes[i][1], hashes[j][1])
            if dist <= threshold:
                duplicates.append({
                    "img1": hashes[i][0],
                    "img2": hashes[j][0],
                    "hamming_distance": dist,
                })
    
    return duplicates


def run_quality_analysis():
    """Run comprehensive quality analysis."""
    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # Load metadata
    if not METADATA_PATH.exists():
        print(f"ERROR: Metadata not found at {METADATA_PATH}")
        return
    
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Total images in metadata: {len(metadata)}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Analyze each image
    print("\n[1/5] Analyzing image quality...")
    results = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        img_path = DATA_PATH / row["path"]
        result = analyze_image(img_path)
        result["person_id"] = row["person_id"]
        result["label"] = row["label"]
        result["source"] = row.get("source", "unknown")
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Save raw analysis
    df.to_csv(OUTPUT_DIR / "quality_raw.csv", index=False)
    
    # Identify issues
    print("\n[2/5] Identifying quality issues...")
    
    issues = {
        "missing": df[~df["exists"]],
        "corrupted": df[df["is_corrupted"]],
        "blank": df[df["is_blank"]],
        "low_contrast": df[df["is_low_contrast"]],
        "too_small": df[df["is_too_small"]],
    }
    
    # Find duplicates (sampling for speed)
    print("\n[3/5] Detecting near-duplicates (sampling 2000 images)...")
    sample_df = df[df["phash"].notna()].sample(min(2000, len(df)))
    duplicates = find_duplicates(sample_df, threshold=3)
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUALITY ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Total images analyzed: {len(df)}")
    print(f"\n⚠️ ISSUES FOUND:")
    print(f"   Missing files:      {len(issues['missing'])}")
    print(f"   Corrupted:          {len(issues['corrupted'])}")
    print(f"   Blank/Empty:        {len(issues['blank'])}")
    print(f"   Low contrast:       {len(issues['low_contrast'])}")
    print(f"   Too small (<500B):  {len(issues['too_small'])}")
    print(f"   Near-duplicates:    {len(duplicates)} pairs (sampled)")
    
    # By source
    print(f"\n📁 BY SOURCE:")
    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        blank_count = source_df["is_blank"].sum()
        low_contrast_count = source_df["is_low_contrast"].sum()
        print(f"   {source}:")
        print(f"      Total: {len(source_df)}")
        print(f"      Blank: {blank_count} ({blank_count/len(source_df)*100:.1f}%)")
        print(f"      Low contrast: {low_contrast_count} ({low_contrast_count/len(source_df)*100:.1f}%)")
    
    # Statistics
    print(f"\n📈 IMAGE STATISTICS:")
    print(f"   Mean pixel value: {df['mean_pixel'].mean():.1f} ± {df['mean_pixel'].std():.1f}")
    print(f"   Mean white ratio: {df['white_ratio'].mean()*100:.1f}%")
    print(f"   Mean signature ratio: {df['signature_ratio'].mean()*100:.1f}%")
    print(f"   Mean contrast: {df['contrast'].mean():.1f}")
    
    # Save issues to files
    print("\n[4/5] Saving issue reports...")
    for issue_name, issue_df in issues.items():
        if len(issue_df) > 0:
            issue_df.to_csv(OUTPUT_DIR / f"issues_{issue_name}.csv", index=False)
            print(f"   Saved: issues_{issue_name}.csv ({len(issue_df)} items)")
    
    if duplicates:
        dup_df = pd.DataFrame(duplicates)
        dup_df.to_csv(OUTPUT_DIR / "issues_duplicates.csv", index=False)
        print(f"   Saved: issues_duplicates.csv ({len(duplicates)} pairs)")
    
    # Create removal candidates
    print("\n[5/5] Creating removal candidates list...")
    
    # Combine all problematic images
    problematic = set()
    for issue_name in ["corrupted", "blank", "too_small"]:
        problematic.update(issues[issue_name]["path"].tolist())
    
    # Add duplicates (keep first, remove second)
    for dup in duplicates:
        problematic.add(dup["img2"])
    
    removal_df = df[df["path"].isin(problematic)]
    removal_df.to_csv(OUTPUT_DIR / "removal_candidates.csv", index=False)
    
    print(f"\n🗑️ REMOVAL CANDIDATES: {len(removal_df)} images")
    print(f"   Saved to: {OUTPUT_DIR / 'removal_candidates.csv'}")
    
    # Final summary
    remaining = len(df) - len(removal_df)
    print(f"\n✅ CLEAN DATASET SIZE: {remaining} images")
    print(f"   Removed: {len(removal_df)} ({len(removal_df)/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return df, removal_df


if __name__ == "__main__":
    run_quality_analysis()
