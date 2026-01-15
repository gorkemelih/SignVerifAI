"""Dataset audit module for SignVerify.

Analyzes dataset quality: person counts, label distribution, corrupted images, duplicates.
Generates JSON and Markdown reports.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

from signverify.config import PathConfig
from signverify.utils.io import write_json, write_markdown
from signverify.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AuditStats:
    """Container for audit statistics."""
    
    total_images: int = 0
    total_persons: int = 0
    genuine_count: int = 0
    forged_count: int = 0
    corrupted_images: list = field(default_factory=list)
    duplicate_groups: dict = field(default_factory=dict)
    size_stats: dict = field(default_factory=dict)
    source_breakdown: dict = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Check if dataset passes all quality checks."""
        return len(self.corrupted_images) == 0


def _get_image_hash(path: Path) -> Optional[str]:
    """Calculate MD5 hash of image file."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def _validate_image(path: Path) -> tuple[bool, Optional[dict]]:
    """
    Validate image and get stats.
    
    Returns:
        Tuple of (is_valid, stats_dict or None)
    """
    try:
        with Image.open(path) as img:
            return True, {
                "width": img.size[0],
                "height": img.size[1],
                "mode": img.mode,
                "format": img.format,
            }
    except Exception as e:
        return False, {"error": str(e)}


def audit_from_metadata(
    metadata_path: Path,
    data_processed_path: Path,
    check_duplicates: bool = True,
) -> AuditStats:
    """
    Audit dataset using existing metadata.csv.
    
    Args:
        metadata_path: Path to metadata.csv
        data_processed_path: Base path for images
        check_duplicates: Whether to check for duplicates (slower)
    
    Returns:
        AuditStats with all statistics
    """
    logger.info("Starting dataset audit from metadata...")
    
    df = pd.read_csv(metadata_path)
    stats = AuditStats()
    
    stats.total_images = len(df)
    stats.total_persons = df["person_id"].nunique()
    stats.genuine_count = len(df[df["label"] == "genuine"])
    stats.forged_count = len(df[df["label"] == "forged"])
    
    # Source breakdown
    stats.source_breakdown = df["source_dataset"].value_counts().to_dict()
    
    # Validate images and collect size stats
    widths, heights = [], []
    hashes: dict[str, list[str]] = defaultdict(list)
    
    for _, row in df.iterrows():
        img_path = data_processed_path / row["path"]
        
        is_valid, img_stats = _validate_image(img_path)
        
        if not is_valid:
            stats.corrupted_images.append(str(img_path))
            logger.warning(f"Corrupted image: {img_path}")
        else:
            widths.append(img_stats["width"])
            heights.append(img_stats["height"])
            
            # Duplicate check
            if check_duplicates:
                img_hash = _get_image_hash(img_path)
                if img_hash:
                    hashes[img_hash].append(str(img_path))
    
    # Size statistics
    if widths:
        stats.size_stats = {
            "width_min": min(widths),
            "width_max": max(widths),
            "height_min": min(heights),
            "height_max": max(heights),
            "width_unique": len(set(widths)),
            "height_unique": len(set(heights)),
        }
    
    # Find duplicates
    if check_duplicates:
        stats.duplicate_groups = {
            h: files for h, files in hashes.items() if len(files) > 1
        }
    
    logger.info(f"Audit complete: {stats.total_images} images, {stats.total_persons} persons")
    logger.info(f"Genuine: {stats.genuine_count}, Forged: {stats.forged_count}")
    logger.info(f"Corrupted: {len(stats.corrupted_images)}, Duplicate groups: {len(stats.duplicate_groups)}")
    
    return stats


def generate_audit_report(stats: AuditStats, output_dir: Path) -> tuple[Path, Path]:
    """
    Generate audit reports in JSON and Markdown formats.
    
    Args:
        stats: AuditStats from audit
        output_dir: Directory to write reports
    
    Returns:
        Tuple of (json_path, markdown_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON report
    json_data = {
        "timestamp": timestamp,
        "total_images": stats.total_images,
        "total_persons": stats.total_persons,
        "genuine_count": stats.genuine_count,
        "forged_count": stats.forged_count,
        "corrupted_count": len(stats.corrupted_images),
        "corrupted_images": stats.corrupted_images,
        "duplicate_groups": len(stats.duplicate_groups),
        "size_stats": stats.size_stats,
        "source_breakdown": stats.source_breakdown,
        "is_healthy": stats.is_healthy,
    }
    
    json_path = output_dir / "audit_report.json"
    write_json(json_data, json_path)
    
    # Markdown report
    md_content = f"""# Dataset Audit Report

**Generated:** {timestamp}

## Summary

| Metric | Value |
|--------|-------|
| Total Images | {stats.total_images:,} |
| Total Persons | {stats.total_persons} |
| Genuine Signatures | {stats.genuine_count:,} |
| Forged Signatures | {stats.forged_count:,} |
| Corrupted Images | {len(stats.corrupted_images)} |
| Duplicate Groups | {len(stats.duplicate_groups)} |

## Source Distribution

| Dataset | Count |
|---------|-------|
"""
    for source, count in stats.source_breakdown.items():
        md_content += f"| {source} | {count:,} |\n"
    
    md_content += f"""
## Image Dimensions

| Metric | Value |
|--------|-------|
| Width Range | {stats.size_stats.get('width_min', 'N/A')} - {stats.size_stats.get('width_max', 'N/A')} px |
| Height Range | {stats.size_stats.get('height_min', 'N/A')} - {stats.size_stats.get('height_max', 'N/A')} px |

## Health Status

{"✅ **HEALTHY** - All quality checks passed" if stats.is_healthy else "❌ **ISSUES FOUND** - See details below"}
"""
    
    if stats.corrupted_images:
        md_content += "\n### Corrupted Images\n\n"
        for img in stats.corrupted_images[:10]:
            md_content += f"- `{img}`\n"
        if len(stats.corrupted_images) > 10:
            md_content += f"\n... and {len(stats.corrupted_images) - 10} more\n"
    
    md_path = output_dir / "audit_report.md"
    write_markdown(md_content, md_path)
    
    logger.info(f"Reports saved: {json_path}, {md_path}")
    
    return json_path, md_path


def run_audit(paths: Optional[PathConfig] = None) -> AuditStats:
    """
    Run full audit pipeline.
    
    Args:
        paths: Path configuration (uses default if None)
    
    Returns:
        AuditStats
    """
    if paths is None:
        paths = PathConfig()
    
    stats = audit_from_metadata(
        metadata_path=paths.metadata_csv,
        data_processed_path=paths.data_processed,
    )
    
    generate_audit_report(stats, paths.reports)
    
    return stats
