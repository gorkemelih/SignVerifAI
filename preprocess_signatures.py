#!/usr/bin/env python3
"""
Signature Preprocessing Pipeline
Converts CEDAR and GPDS datasets to standardized format:
- Grayscale conversion
- Auto-crop with signature detection
- 224x224 resize
- Background cleanup
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path("/Users/melihozcan/Desktop/TUBITAK 2209-A")
DATA_RAW = BASE_DIR / "data_raw"
DATA_PROCESSED = BASE_DIR / "data_processed"
OUTPUT_SIZE = (224, 224)
MARGIN_RATIO = 0.05  # 5% margin around signature


def ensure_grayscale(img):
    """Convert image to grayscale if needed."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def clean_background(img):
    """
    Clean background while preserving signature strokes.
    Uses adaptive thresholding to handle varying lighting conditions.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Adaptive thresholding for better stroke preservation
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        21, 10
    )
    
    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Invert back (white background, black signature)
    result = 255 - cleaned
    
    return result


def auto_crop(img, margin_ratio=MARGIN_RATIO):
    """
    Automatically crop to signature bounds with safe margin.
    """
    # Threshold to find signature
    if img.max() > 200:  # White background
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img  # Return original if no contours found
    
    # Get bounding box of all contours
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add margin
    margin_x = int((x_max - x_min) * margin_ratio)
    margin_y = int((y_max - y_min) * margin_ratio)
    
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(img.shape[1], x_max + margin_x)
    y_max = min(img.shape[0], y_max + margin_y)
    
    # Crop
    cropped = img[y_min:y_max, x_min:x_max]
    
    # Ensure we got a valid crop
    if cropped.size == 0:
        return img
    
    return cropped


def resize_with_padding(img, target_size=OUTPUT_SIZE):
    """
    Resize image to target size while maintaining aspect ratio.
    Adds white padding to fill remaining space.
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to fit within target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create white canvas
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    
    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def preprocess_image(input_path, output_path):
    """
    Full preprocessing pipeline for a single image.
    """
    try:
        # Read image
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"Failed to read: {input_path}")
            return None
        
        # 1. Convert to grayscale
        gray = ensure_grayscale(img)
        
        # 2. Clean background
        cleaned = clean_background(gray)
        
        # 3. Auto-crop
        cropped = auto_crop(cleaned)
        
        # 4. Resize with padding
        final = resize_with_padding(cropped)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), final)
        
        return output_path
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return None


def process_cedar():
    """Process CEDAR dataset."""
    logger.info("Processing CEDAR dataset...")
    
    cedar_path = DATA_RAW / "CEDAR" / "signatures"
    records = []
    
    # Get all persons from genuine folder
    genuine_path = cedar_path / "full_org"
    forged_path = cedar_path / "full_forg"
    
    # Extract unique person IDs
    persons = set()
    for f in genuine_path.glob("*.png"):
        parts = f.stem.split('_')
        if len(parts) >= 2:
            persons.add(int(parts[1]))
    
    persons = sorted(persons)
    logger.info(f"Found {len(persons)} persons in CEDAR")
    
    # Process each person
    for person_id in persons:
        # Standardize person ID format (CEDAR uses different numbering)
        person_folder = DATA_PROCESSED / "images" / f"person_{person_id:04d}"
        
        # Process genuine signatures
        genuine_count = 0
        for f in genuine_path.glob(f"original_{person_id}_*.png"):
            genuine_count += 1
            output_path = person_folder / f"genuine_{genuine_count:04d}.png"
            result = preprocess_image(f, output_path)
            if result:
                records.append({
                    'path': str(output_path.relative_to(DATA_PROCESSED)),
                    'person_id': f"{person_id:04d}",
                    'label': 'genuine',
                    'source_dataset': 'CEDAR'
                })
        
        # Process forged signatures
        forged_count = 0
        for f in forged_path.glob(f"forgeries_{person_id}_*.png"):
            forged_count += 1
            output_path = person_folder / f"forged_{forged_count:04d}.png"
            result = preprocess_image(f, output_path)
            if result:
                records.append({
                    'path': str(output_path.relative_to(DATA_PROCESSED)),
                    'person_id': f"{person_id:04d}",
                    'label': 'forged',
                    'source_dataset': 'CEDAR'
                })
    
    logger.info(f"CEDAR: Processed {len(records)} images")
    return records


def process_gpds():
    """Process GPDS dataset."""
    logger.info("Processing GPDS dataset...")
    
    gpds_path = DATA_RAW / "GPDS" / "New folder (10)"
    records = []
    
    # Get all person folders
    persons = set()
    for split in ['train', 'test']:
        split_path = gpds_path / split
        if split_path.exists():
            for person_dir in split_path.iterdir():
                if person_dir.is_dir():
                    try:
                        persons.add(int(person_dir.name))
                    except ValueError:
                        pass
    
    persons = sorted(persons)
    logger.info(f"Found {len(persons)} persons in GPDS")
    
    # Offset person IDs to avoid collision with CEDAR (55 persons)
    person_offset = 100
    
    for person_id in persons:
        # Standardize person ID format with offset
        new_person_id = person_offset + person_id
        person_folder = DATA_PROCESSED / "images" / f"person_{new_person_id:04d}"
        
        genuine_count = 0
        forged_count = 0
        
        # Process from both train and test
        for split in ['train', 'test']:
            person_path = gpds_path / split / str(person_id)
            if not person_path.exists():
                continue
            
            # Process genuine signatures
            genuine_path = person_path / "genuine"
            if genuine_path.exists():
                for f in genuine_path.glob("*.jpg"):
                    genuine_count += 1
                    output_path = person_folder / f"genuine_{genuine_count:04d}.png"
                    result = preprocess_image(f, output_path)
                    if result:
                        records.append({
                            'path': str(output_path.relative_to(DATA_PROCESSED)),
                            'person_id': f"{new_person_id:04d}",
                            'label': 'genuine',
                            'source_dataset': 'GPDS'
                        })
            
            # Process forged signatures
            forge_path = person_path / "forge"
            if forge_path.exists():
                for f in forge_path.glob("*.jpg"):
                    forged_count += 1
                    output_path = person_folder / f"forged_{forged_count:04d}.png"
                    result = preprocess_image(f, output_path)
                    if result:
                        records.append({
                            'path': str(output_path.relative_to(DATA_PROCESSED)),
                            'person_id': f"{new_person_id:04d}",
                            'label': 'forged',
                            'source_dataset': 'GPDS'
                        })
    
    logger.info(f"GPDS: Processed {len(records)} images")
    return records


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Signature Preprocessing Pipeline")
    logger.info("=" * 60)
    
    # Create output directory
    (DATA_PROCESSED / "images").mkdir(parents=True, exist_ok=True)
    
    # Process both datasets
    cedar_records = process_cedar()
    gpds_records = process_gpds()
    
    # Combine records
    all_records = cedar_records + gpds_records
    
    # Create metadata CSV
    df = pd.DataFrame(all_records)
    metadata_path = DATA_PROCESSED / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total images processed: {len(all_records)}")
    logger.info(f"  - CEDAR: {len(cedar_records)}")
    logger.info(f"  - GPDS: {len(gpds_records)}")
    logger.info(f"Metadata saved to: {metadata_path}")
    
    # Print summary
    print("\n📊 Label Distribution:")
    print(df['label'].value_counts())
    print("\n📊 Source Distribution:")
    print(df['source_dataset'].value_counts())
    print(f"\n📊 Unique Persons: {df['person_id'].nunique()}")


if __name__ == "__main__":
    main()
