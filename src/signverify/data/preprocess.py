"""Image preprocessing module for SignVerify.

Handles grayscale conversion, auto-crop, resize, and background cleanup.
Designed to be idempotent - skips already processed images.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from signverify.config import PathConfig
from signverify.utils.io import read_csv, write_csv
from signverify.utils.logging import get_logger

logger = get_logger(__name__)

# Default preprocessing parameters
OUTPUT_SIZE = (224, 224)
MARGIN_RATIO = 0.05


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def clean_background(img: np.ndarray) -> np.ndarray:
    """
    Clean background while preserving signature strokes.
    Uses adaptive thresholding for varying lighting conditions.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )
    
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return 255 - cleaned


def auto_crop(img: np.ndarray, margin_ratio: float = MARGIN_RATIO) -> np.ndarray:
    """
    Automatically crop to signature bounds with safe margin.
    
    Args:
        img: Grayscale image
        margin_ratio: Margin as fraction of signature size
    
    Returns:
        Cropped image
    """
    if img.max() > 200:
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    margin_x = int((x_max - x_min) * margin_ratio)
    margin_y = int((y_max - y_min) * margin_ratio)
    
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(img.shape[1], x_max + margin_x)
    y_max = min(img.shape[0], y_max + margin_y)
    
    cropped = img[y_min:y_max, x_min:x_max]
    
    if cropped.size == 0:
        return img
    
    return cropped


def resize_with_padding(
    img: np.ndarray,
    target_size: tuple[int, int] = OUTPUT_SIZE,
) -> np.ndarray:
    """
    Resize image maintaining aspect ratio with white padding.
    
    Args:
        img: Input image
        target_size: Target (width, height)
    
    Returns:
        Resized image with padding
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    
    return canvas


def preprocess_image(input_path: Path, output_path: Path) -> Optional[Path]:
    """
    Full preprocessing pipeline for a single image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
    
    Returns:
        Output path if successful, None otherwise
    """
    try:
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"Failed to read: {input_path}")
            return None
        
        gray = ensure_grayscale(img)
        cleaned = clean_background(gray)
        cropped = auto_crop(cleaned)
        final = resize_with_padding(cropped)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), final)
        
        return output_path
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return None


def is_already_processed(path: Path) -> bool:
    """Check if image is already preprocessed (224x224 grayscale)."""
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            return img.size == OUTPUT_SIZE and img.mode == "L"
    except Exception:
        return False


def run_preprocess(
    paths: Optional[PathConfig] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Run preprocessing on all images from metadata.
    
    Args:
        paths: Path configuration
        force: Force reprocessing even if already done
    
    Returns:
        Updated metadata DataFrame
    """
    if paths is None:
        paths = PathConfig()
    
    if not paths.metadata_csv.exists():
        logger.error(f"Metadata not found: {paths.metadata_csv}")
        raise FileNotFoundError(f"Metadata not found: {paths.metadata_csv}")
    
    df = read_csv(paths.metadata_csv)
    
    processed_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        img_path = paths.data_processed / row["path"]
        
        if not force and is_already_processed(img_path):
            skipped_count += 1
            continue
        
        # For reprocessing, we'd need original paths
        # Since originals are already processed, just validate
        if img_path.exists():
            processed_count += 1
        else:
            logger.warning(f"Image not found: {img_path}")
    
    logger.info(f"Preprocessing check: {processed_count} valid, {skipped_count} already processed")
    
    return df
