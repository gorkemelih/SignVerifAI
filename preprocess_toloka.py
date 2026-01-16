#!/usr/bin/env python3
"""
Preprocess Toloka WhoSigned dataset and integrate with existing datasets.

This script:
1. Processes Toloka dataset (real/forged signatures)
2. Resizes to 224x224
3. Normalizes and cleans images
4. Creates metadata
5. Merges with existing CEDAR+GDPS metadata
"""

import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict

# Configuration
TOLOKA_RAW_PATH = Path("data_raw/toloka/data/data")
OUTPUT_PATH = Path("data_processed")
TARGET_SIZE = (224, 224)
DATASET_PREFIX = "toloka"


def clean_and_resize_image(img_path: Path, target_size: tuple) -> Image.Image:
    """Load, clean, and resize signature image."""
    try:
        img = Image.open(img_path)
        
        # Convert to grayscale
        if img.mode != "L":
            img = img.convert("L")
        
        # Convert to numpy for processing
        arr = np.array(img, dtype=np.float32)
        
        # Normalize to 0-255
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        arr = arr.astype(np.uint8)
        
        # Create image
        img = Image.fromarray(arr, mode="L")
        
        # Resize maintaining aspect ratio with padding
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create white background and paste centered
        new_img = Image.new("L", target_size, 255)
        paste_x = (target_size[0] - img.size[0]) // 2
        paste_y = (target_size[1] - img.size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def get_person_id_from_folder(folder_name: str) -> str:
    """Extract person ID from Toloka folder name."""
    # Folder names are like: "1140762B-8D0C-4456-A290-63A82666C69E.jpg"
    # We use the UUID part as person_id
    person_id = folder_name.replace(".jpg", "").replace(" ", "_")
    # Clean up and shorten if needed
    if len(person_id) > 50:
        person_id = person_id[:50]
    return person_id


def process_toloka_dataset():
    """Process Toloka dataset and create metadata."""
    print("=" * 60)
    print("PROCESSING TOLOKA WHOSIGNED DATASET")
    print("=" * 60)
    
    metadata_records = []
    person_counter = defaultdict(int)
    processed_count = 0
    error_count = 0
    
    # Process real (genuine) signatures
    real_path = TOLOKA_RAW_PATH / "real"
    forged_path = TOLOKA_RAW_PATH / "forged"
    
    # Create output directories
    (OUTPUT_PATH / DATASET_PREFIX).mkdir(parents=True, exist_ok=True)
    
    # Map folder names to person IDs
    folder_to_person = {}
    person_id_counter = 1
    
    # First pass: identify unique persons from real signatures
    print("\n[1/4] Identifying unique persons from real signatures...")
    if real_path.exists():
        for folder in sorted(real_path.iterdir()):
            if folder.is_dir() and folder.name.endswith(".jpg"):
                base_name = folder.name.replace(".jpg", "")
                if base_name not in folder_to_person:
                    folder_to_person[base_name] = f"{DATASET_PREFIX}_{person_id_counter:04d}"
                    person_id_counter += 1
    
    # Also check forged folders for additional persons
    print("[2/4] Checking forged folder for additional persons...")
    if forged_path.exists():
        for folder in sorted(forged_path.iterdir()):
            if folder.is_dir() and folder.name.endswith(".jpg"):
                base_name = folder.name.replace(".jpg", "")
                if base_name not in folder_to_person:
                    folder_to_person[base_name] = f"{DATASET_PREFIX}_{person_id_counter:04d}"
                    person_id_counter += 1
    
    print(f"   Found {len(folder_to_person)} unique persons")
    
    # Process real signatures
    print("\n[3/4] Processing REAL (genuine) signatures...")
    if real_path.exists():
        folders = [f for f in real_path.iterdir() if f.is_dir()]
        for folder in folders:
            base_name = folder.name.replace(".jpg", "")
            person_id = folder_to_person.get(base_name, f"{DATASET_PREFIX}_unknown")
            
            # Get all images in folder
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpg.jpg"))
            
            for idx, img_path in enumerate(images):
                # Create output filename
                output_filename = f"{person_id}_genuine_{idx:03d}.png"
                output_path = OUTPUT_PATH / DATASET_PREFIX / output_filename
                
                # Process image
                processed_img = clean_and_resize_image(img_path, TARGET_SIZE)
                if processed_img:
                    processed_img.save(output_path)
                    
                    metadata_records.append({
                        "path": f"{DATASET_PREFIX}/{output_filename}",
                        "person_id": person_id,
                        "label": "genuine",
                        "source": "toloka",
                        "original_file": img_path.name,
                    })
                    processed_count += 1
                else:
                    error_count += 1
        
        print(f"   Processed {processed_count} genuine signatures")
    
    # Process forged signatures  
    print("\n[4/4] Processing FORGED signatures...")
    forged_processed = 0
    if forged_path.exists():
        folders = [f for f in forged_path.iterdir() if f.is_dir()]
        for folder in folders:
            base_name = folder.name.replace(".jpg", "")
            person_id = folder_to_person.get(base_name, f"{DATASET_PREFIX}_unknown")
            
            # Get all images in folder
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpg.jpg"))
            
            for idx, img_path in enumerate(images):
                # Create output filename
                output_filename = f"{person_id}_forged_{idx:03d}.png"
                output_path = OUTPUT_PATH / DATASET_PREFIX / output_filename
                
                # Process image
                processed_img = clean_and_resize_image(img_path, TARGET_SIZE)
                if processed_img:
                    processed_img.save(output_path)
                    
                    metadata_records.append({
                        "path": f"{DATASET_PREFIX}/{output_filename}",
                        "person_id": person_id,
                        "label": "forged",
                        "source": "toloka",
                        "original_file": img_path.name,
                    })
                    forged_processed += 1
                    processed_count += 1
                else:
                    error_count += 1
        
        print(f"   Processed {forged_processed} forged signatures")
    
    # Create Toloka metadata
    toloka_metadata = pd.DataFrame(metadata_records)
    toloka_metadata_path = OUTPUT_PATH / "metadata_toloka.csv"
    toloka_metadata.to_csv(toloka_metadata_path, index=False)
    print(f"\n✓ Toloka metadata saved: {toloka_metadata_path}")
    
    # Load existing metadata and merge
    print("\n[MERGING] Combining with existing datasets...")
    existing_metadata_path = OUTPUT_PATH / "metadata.csv"
    
    if existing_metadata_path.exists():
        existing_metadata = pd.read_csv(existing_metadata_path)
        
        # Check if toloka already in metadata
        if "source" in existing_metadata.columns:
            existing_metadata = existing_metadata[existing_metadata["source"] != "toloka"]
        
        # Add source column if not exists
        if "source" not in existing_metadata.columns:
            existing_metadata["source"] = "cedar_gdps"
        
        # Merge
        combined_metadata = pd.concat([existing_metadata, toloka_metadata], ignore_index=True)
    else:
        combined_metadata = toloka_metadata
    
    # Save combined metadata
    combined_metadata.to_csv(existing_metadata_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)
    print(f"Toloka processed: {processed_count} images")
    print(f"Errors: {error_count}")
    print(f"\nCombined dataset:")
    print(f"  Total images: {len(combined_metadata)}")
    print(f"  Unique persons: {combined_metadata['person_id'].nunique()}")
    print(f"\n  By label:")
    print(combined_metadata["label"].value_counts().to_string())
    print(f"\n  By source:")
    print(combined_metadata["source"].value_counts().to_string())
    print("=" * 60)
    
    return combined_metadata


if __name__ == "__main__":
    process_toloka_dataset()
