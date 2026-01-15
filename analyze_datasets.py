#!/usr/bin/env python3
"""
Dataset Analysis Script for CEDAR and GPDS Signature Datasets
Analyzes: person counts, signature counts, image formats, sizes, corrupted files, duplicates
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
import json

BASE_DIR = Path("/Users/melihozcan/Desktop/TUBITAK 2209-A")
DATA_RAW = BASE_DIR / "data_raw"

def get_image_hash(filepath):
    """Calculate MD5 hash of image file for duplicate detection."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def analyze_image(filepath):
    """Analyze single image and return stats."""
    try:
        with Image.open(filepath) as img:
            return {
                'valid': True,
                'format': img.format,
                'mode': img.mode,
                'width': img.size[0],
                'height': img.size[1],
                'size_bytes': os.path.getsize(filepath)
            }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def analyze_cedar():
    """Analyze CEDAR dataset."""
    print("\n" + "="*60)
    print("CEDAR Dataset Analysis")
    print("="*60)
    
    cedar_path = DATA_RAW / "CEDAR" / "signatures"
    
    # Paths
    genuine_path = cedar_path / "full_org"
    forged_path = cedar_path / "full_forg"
    
    results = {
        'genuine': {'files': [], 'persons': set()},
        'forged': {'files': [], 'persons': set()}
    }
    
    all_stats = []
    corrupted = []
    hashes = defaultdict(list)
    
    # Analyze genuine signatures
    if genuine_path.exists():
        for f in genuine_path.glob("*.png"):
            # Extract person ID from filename: original_X_Y.png
            parts = f.stem.split('_')
            if len(parts) >= 2:
                person_id = parts[1]
                results['genuine']['persons'].add(person_id)
            results['genuine']['files'].append(f)
            
            stats = analyze_image(f)
            stats['path'] = str(f)
            stats['type'] = 'genuine'
            all_stats.append(stats)
            
            if not stats['valid']:
                corrupted.append(str(f))
            else:
                h = get_image_hash(f)
                if h:
                    hashes[h].append(str(f))
    
    # Analyze forged signatures
    if forged_path.exists():
        for f in forged_path.glob("*.png"):
            # Extract person ID from filename: forgeries_X_Y.png
            parts = f.stem.split('_')
            if len(parts) >= 2:
                person_id = parts[1]
                results['forged']['persons'].add(person_id)
            results['forged']['files'].append(f)
            
            stats = analyze_image(f)
            stats['path'] = str(f)
            stats['type'] = 'forged'
            all_stats.append(stats)
            
            if not stats['valid']:
                corrupted.append(str(f))
            else:
                h = get_image_hash(f)
                if h:
                    hashes[h].append(str(f))
    
    # Calculate size distribution
    valid_stats = [s for s in all_stats if s['valid']]
    widths = [s['width'] for s in valid_stats]
    heights = [s['height'] for s in valid_stats]
    
    # Find duplicates
    duplicates = {h: files for h, files in hashes.items() if len(files) > 1}
    
    # Print results
    all_persons = results['genuine']['persons'].union(results['forged']['persons'])
    print(f"\n📊 Person Count: {len(all_persons)}")
    print(f"   Persons with genuine signatures: {len(results['genuine']['persons'])}")
    print(f"   Persons with forged signatures: {len(results['forged']['persons'])}")
    
    print(f"\n📝 Signature Counts:")
    print(f"   Genuine signatures: {len(results['genuine']['files'])}")
    print(f"   Forged signatures: {len(results['forged']['files'])}")
    print(f"   Total: {len(results['genuine']['files']) + len(results['forged']['files'])}")
    
    print(f"\n🖼️  Image Format: PNG")
    if valid_stats:
        modes = set(s['mode'] for s in valid_stats)
        print(f"   Color modes: {modes}")
        print(f"   Width range: {min(widths)} - {max(widths)} px")
        print(f"   Height range: {min(heights)} - {max(heights)} px")
        avg_size = sum(s['size_bytes'] for s in valid_stats) / len(valid_stats)
        print(f"   Average file size: {avg_size/1024:.1f} KB")
    
    print(f"\n❌ Corrupted Images: {len(corrupted)}")
    for c in corrupted[:5]:
        print(f"   - {c}")
    
    print(f"\n🔄 Duplicate Images: {len(duplicates)} groups")
    for h, files in list(duplicates.items())[:3]:
        print(f"   - {len(files)} files with same hash")
    
    return {
        'dataset': 'CEDAR',
        'persons': len(all_persons),
        'genuine_count': len(results['genuine']['files']),
        'forged_count': len(results['forged']['files']),
        'corrupted': corrupted,
        'duplicates': duplicates
    }

def analyze_gpds():
    """Analyze GPDS dataset."""
    print("\n" + "="*60)
    print("GPDS Dataset Analysis")
    print("="*60)
    
    gpds_path = DATA_RAW / "GPDS" / "New folder (10)"
    
    results = {
        'genuine': {'files': [], 'persons': set()},
        'forged': {'files': [], 'persons': set()}
    }
    
    all_stats = []
    corrupted = []
    hashes = defaultdict(list)
    
    # Process both train and test
    for split in ['train', 'test']:
        split_path = gpds_path / split
        if not split_path.exists():
            continue
            
        # Process each person folder
        for person_dir in split_path.iterdir():
            if not person_dir.is_dir():
                continue
            person_id = person_dir.name
            
            # Genuine
            genuine_path = person_dir / "genuine"
            if genuine_path.exists():
                for f in genuine_path.glob("*.jpg"):
                    results['genuine']['persons'].add(person_id)
                    results['genuine']['files'].append(f)
                    
                    stats = analyze_image(f)
                    stats['path'] = str(f)
                    stats['type'] = 'genuine'
                    stats['split'] = split
                    all_stats.append(stats)
                    
                    if not stats['valid']:
                        corrupted.append(str(f))
                    else:
                        h = get_image_hash(f)
                        if h:
                            hashes[h].append(str(f))
            
            # Forged
            forge_path = person_dir / "forge"
            if forge_path.exists():
                for f in forge_path.glob("*.jpg"):
                    results['forged']['persons'].add(person_id)
                    results['forged']['files'].append(f)
                    
                    stats = analyze_image(f)
                    stats['path'] = str(f)
                    stats['type'] = 'forged'
                    stats['split'] = split
                    all_stats.append(stats)
                    
                    if not stats['valid']:
                        corrupted.append(str(f))
                    else:
                        h = get_image_hash(f)
                        if h:
                            hashes[h].append(str(f))
    
    # Calculate size distribution
    valid_stats = [s for s in all_stats if s['valid']]
    widths = [s['width'] for s in valid_stats] if valid_stats else [0]
    heights = [s['height'] for s in valid_stats] if valid_stats else [0]
    
    # Find duplicates
    duplicates = {h: files for h, files in hashes.items() if len(files) > 1}
    
    # Print results
    all_persons = results['genuine']['persons'].union(results['forged']['persons'])
    print(f"\n📊 Person Count: {len(all_persons)}")
    print(f"   Persons with genuine signatures: {len(results['genuine']['persons'])}")
    print(f"   Persons with forged signatures: {len(results['forged']['persons'])}")
    
    print(f"\n📝 Signature Counts:")
    print(f"   Genuine signatures: {len(results['genuine']['files'])}")
    print(f"   Forged signatures: {len(results['forged']['files'])}")
    print(f"   Total: {len(results['genuine']['files']) + len(results['forged']['files'])}")
    
    # Split breakdown
    train_count = len([s for s in all_stats if s.get('split') == 'train'])
    test_count = len([s for s in all_stats if s.get('split') == 'test'])
    print(f"   Train split: {train_count}")
    print(f"   Test split: {test_count}")
    
    print(f"\n🖼️  Image Format: JPEG")
    if valid_stats:
        modes = set(s['mode'] for s in valid_stats)
        print(f"   Color modes: {modes}")
        print(f"   Width range: {min(widths)} - {max(widths)} px")
        print(f"   Height range: {min(heights)} - {max(heights)} px")
        avg_size = sum(s['size_bytes'] for s in valid_stats) / len(valid_stats)
        print(f"   Average file size: {avg_size/1024:.1f} KB")
    
    print(f"\n❌ Corrupted Images: {len(corrupted)}")
    for c in corrupted[:5]:
        print(f"   - {c}")
    
    print(f"\n🔄 Duplicate Images: {len(duplicates)} groups")
    for h, files in list(duplicates.items())[:3]:
        print(f"   - {len(files)} files with same hash")
    
    return {
        'dataset': 'GPDS',
        'persons': len(all_persons),
        'genuine_count': len(results['genuine']['files']),
        'forged_count': len(results['forged']['files']),
        'corrupted': corrupted,
        'duplicates': duplicates
    }

if __name__ == "__main__":
    print("🔍 Signature Dataset Analysis Tool")
    print("="*60)
    
    cedar_results = analyze_cedar()
    gpds_results = analyze_gpds()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nCEDAR: {cedar_results['persons']} persons, {cedar_results['genuine_count']} genuine, {cedar_results['forged_count']} forged")
    print(f"GPDS:  {gpds_results['persons']} persons, {gpds_results['genuine_count']} genuine, {gpds_results['forged_count']} forged")
    print(f"\nTotal: {cedar_results['persons'] + gpds_results['persons']} persons")
    print(f"Total Genuine: {cedar_results['genuine_count'] + gpds_results['genuine_count']}")
    print(f"Total Forged: {cedar_results['forged_count'] + gpds_results['forged_count']}")
    print(f"Total Images: {cedar_results['genuine_count'] + cedar_results['forged_count'] + gpds_results['genuine_count'] + gpds_results['forged_count']}")
