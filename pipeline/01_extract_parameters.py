#!/usr/bin/env python3
"""
Script to extract parameters from cached files and save them as JSON files.
This script will:
1. Find all cache folders (cache_AL1, cache_AL2, etc.)
2. Extract parameters from processed_binary_mask files
3. Extract peak_min_distance from peak_markers files if available
4. Save parameters as parameters.json in each cache folder
"""

import os
import re
import json
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bombusal.config import data_root


def extract_parameters_from_filename(filename, file_type="processed_binary_mask"):
    """
    Extract parameters from cached filename.
    
    Args:
        filename (str): The filename to parse
        file_type (str): Type of file - 'processed_binary_mask', 'peak_markers', or 'binary_mask'
    
    Returns:
        dict: Extracted parameters or None if parsing failed
    """
    
    if file_type == "processed_binary_mask":
        # Format: processed_binary_mask_C80-99_D0.5_B0.5.tif
        pattern = r"processed_binary_mask_C(\d+)-(\d+)_D([0-9.]+)_B([0-9.]+)\.tif"
        match = re.match(pattern, filename)
        if match:
            return {
                "lower_percentile": int(match.group(1)),
                "upper_percentile": int(match.group(2)),
                "gaussian_sigma": float(match.group(3)),
                "threshold_factor": float(match.group(4))
            }
    
    elif file_type == "peak_markers":
        # Format: peak_markers_C80-99_D0.5_B0.5_DT60_PM35.tif (new format with DT)
        # Also support old format: peak_markers_C80-99_D0.5_B0.5_PM35.tif (without DT)
        pattern_new = r"peak_markers_C(\d+)-(\d+)_D([0-9.]+)_B([0-9.]+)_DT([0-9.]+)_PM([0-9.]+)\.tif"
        pattern_old = r"peak_markers_C(\d+)-(\d+)_D([0-9.]+)_B([0-9.]+)_PM([0-9.]+)\.tif"
        
        match = re.match(pattern_new, filename)
        if match:
            return {
                "lower_percentile": int(match.group(1)),
                "upper_percentile": int(match.group(2)),
                "gaussian_sigma": float(match.group(3)),
                "threshold_factor": float(match.group(4)),
                "distance_threshold": int(match.group(5)),
                "peak_min_distance": int(match.group(6))
            }
        else:
            # Try old format without DT parameter
            match = re.match(pattern_old, filename)
            if match:
                return {
                    "lower_percentile": int(match.group(1)),
                    "upper_percentile": int(match.group(2)),
                    "gaussian_sigma": float(match.group(3)),
                    "threshold_factor": float(match.group(4)),
                    "peak_min_distance": int(match.group(5))
                    # distance_threshold will use default value
                }
    
    elif file_type == "binary_mask":
        # Format: binary_mask_C93-98_D0.5_B0.5.tif (for cache_wb)
        pattern = r"binary_mask_C(\d+)-(\d+)_D([0-9.]+)_B([0-9.]+)\.tif"
        match = re.match(pattern, filename)
        if match:
            return {
                "lower_percentile": int(match.group(1)),
                "upper_percentile": int(match.group(2)),
                "gaussian_sigma": float(match.group(3)),
                "threshold_factor": float(match.group(4))
            }
    
    return None


def find_cache_folders(root_path):
    """
    Find all cache folders in the given root path.
    
    Args:
        root_path (Path): Root path to search for cache folders
    
    Returns:
        list: List of cache folder paths
    """
    cache_folders = []
    
    # Search for cache folders recursively
    for item in root_path.rglob("cache_*"):
        if item.is_dir():
            cache_folders.append(item)
    
    return cache_folders


def process_cache_folder(cache_folder):
    """
    Process a single cache folder to extract parameters.
    
    Args:
        cache_folder (Path): Path to the cache folder
    
    Returns:
        tuple: (parameters_dict or None, warnings_list)
    """
    print(f"\nProcessing cache folder: {cache_folder}")
    
    # Initialize parameters with defaults
    parameters = {
        "global_mean": 5685.8,
        "global_std": 8127.3,
        "lower_percentile": None,
        "upper_percentile": None,
        "gaussian_sigma": None,
        "threshold_factor": None,
        "peak_min_distance": None,
        "distance_threshold": 60,  # Default value for distance threshold
        "source_files": {
            "processed_binary_mask": None,
            "peak_markers": None,
            "binary_mask": None
        }
    }
    
    warnings = []
    cache_type = None
    
    # Determine cache type based on folder name
    if cache_folder.name.startswith('cache_AL'):
        cache_type = "AL"
    elif cache_folder.name.startswith('cache_wb') or cache_folder.name == 'cache_wb':
        cache_type = "wb"
    else:
        cache_type = "unknown"
    
    print(f"  Cache type: {cache_type}")
    
    if cache_type == "wb":
        # For cache_wb, only look for binary_mask files
        binary_mask_files = list(cache_folder.glob("binary_mask_C*-*_D*_B*.tif"))
        
        if len(binary_mask_files) == 0:
            print("  No binary_mask files found in cache_wb")
            return None, warnings
        elif len(binary_mask_files) > 1:
            warning_msg = f"Multiple binary_mask files found in {cache_folder}: {[f.name for f in binary_mask_files]}"
            print(f"  WARNING: {warning_msg}")
            warnings.append(warning_msg)
            return None, warnings
        else:
            # Exactly one binary_mask file
            binary_mask_file = binary_mask_files[0]
            print(f"  Found binary_mask: {binary_mask_file.name}")
            
            mask_params = extract_parameters_from_filename(binary_mask_file.name, "binary_mask")
            if mask_params:
                parameters.update(mask_params)
                parameters["source_files"]["binary_mask"] = binary_mask_file.name
                parameters["peak_min_distance"] = 35  # Use default for wb
                print(f"  Extracted from binary_mask: {mask_params}")
                return parameters, warnings
            else:
                print(f"  Failed to parse binary_mask filename: {binary_mask_file.name}")
                return None, warnings
    
    elif cache_type == "AL":
        # For cache_AL*, look for processed_binary_mask and peak_markers files
        
        # Check processed_binary_mask files
        mask_files = list(cache_folder.glob("processed_binary_mask_C*-*_D*_B*.tif"))
        mask_params = None
        
        if len(mask_files) == 0:
            print("  No processed_binary_mask files found")
        elif len(mask_files) > 1:
            warning_msg = f"Multiple processed_binary_mask files found in {cache_folder}: {[f.name for f in mask_files]}"
            print(f"  WARNING: {warning_msg}")
            warnings.append(warning_msg)
        else:
            # Exactly one processed_binary_mask file
            mask_file = mask_files[0]
            print(f"  Found processed_binary_mask: {mask_file.name}")
            
            mask_params = extract_parameters_from_filename(mask_file.name, "processed_binary_mask")
            if mask_params:
                parameters.update(mask_params)
                parameters["source_files"]["processed_binary_mask"] = mask_file.name
                print(f"  Extracted from mask: {mask_params}")
            else:
                print(f"  Failed to parse mask filename: {mask_file.name}")
        
        # Check peak_markers files (support both new and old formats)
        marker_files_old = list(cache_folder.glob("peak_markers_C*-*_D*_B*_PM*.tif"))
        marker_files_new = list(cache_folder.glob("peak_markers_C*-*_D*_B*_DT*_PM*.tif"))
        # Remove duplicates (new format files might match both patterns)
        marker_files = list(set(marker_files_old + marker_files_new))
        
        if len(marker_files) == 0:
            default_peak_min_distance = 20
            print(f"  No peak_markers files found - using default peak_min_distance: {default_peak_min_distance}")
            parameters["peak_min_distance"] = default_peak_min_distance  # Default value
        elif len(marker_files) > 1:
            warning_msg = f"Multiple peak_markers files found in {cache_folder}: {[f.name for f in marker_files]}"
            print(f"  WARNING: {warning_msg}")
            warnings.append(warning_msg)
            # Still use default if multiple files
            parameters["peak_min_distance"] = 35
        else:
            # Exactly one peak_markers file
            marker_file = marker_files[0]
            print(f"  Found peak_markers: {marker_file.name}")
            
            marker_params = extract_parameters_from_filename(marker_file.name, "peak_markers")
            if marker_params:
                # Update peak_min_distance from markers file
                if "peak_min_distance" in marker_params:
                    parameters["peak_min_distance"] = marker_params["peak_min_distance"]
                    parameters["source_files"]["peak_markers"] = marker_file.name
                    print(f"  Extracted peak_min_distance: {marker_params['peak_min_distance']}")
                
                # Update distance_threshold if present in new format files
                if "distance_threshold" in marker_params:
                    parameters["distance_threshold"] = marker_params["distance_threshold"]
                    print(f"  Extracted distance_threshold: {marker_params['distance_threshold']}")
            else:
                print(f"  Failed to parse marker filename: {marker_file.name}")
                parameters["peak_min_distance"] = 35  # Use default
        
        # Check if we got the essential parameters (from mask file)
        if mask_params and parameters["lower_percentile"] is not None:
            return parameters, warnings
        else:
            print(f"  Could not extract essential parameters from {cache_folder}")
            return None, warnings
    
    else:
        print(f"  Unknown cache type: {cache_type}")
        return None, warnings


def save_parameters_json(cache_folder, parameters):
    """
    Save parameters to JSON file in the cache folder.
    
    Args:
        cache_folder (Path): Path to the cache folder
        parameters (dict): Parameters to save
    """
    json_file = cache_folder / "parameters.json"
    
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(parameters, f, indent=2, ensure_ascii=False)
        print(f"  Saved parameters to: {json_file}")
        return True
    except Exception as e:
        print(f"  Error saving parameters to {json_file}: {e}")
        return False


def save_warnings_report(root_path, all_warnings):
    """
    Save warnings report to a file.
    
    Args:
        root_path (Path): Root path where the report will be saved
        all_warnings (list): List of all warnings
    """
    if not all_warnings:
        return
    
    report_file = root_path / "cache_extraction_warnings.txt"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Cache Parameter Extraction Warnings Report\n")
            f.write("=" * 50 + "\n\n")
            
            for i, warning in enumerate(all_warnings, 1):
                f.write(f"{i}. {warning}\n")
        
        print(f"\nWarnings report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error saving warnings report: {e}")


def save_failed_folders_report(root_path, failed_folders):
    """
    Save failed folders report to a file.
    
    Args:
        root_path (Path): Root path where the report will be saved
        failed_folders (list): List of failed cache folder paths
    """
    if not failed_folders:
        return
    
    report_file = root_path / "cache_extraction_failures.txt"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Cache Parameter Extraction Failures Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("The following cache folders failed to extract parameters:\n")
            f.write("(Usually due to missing processed_binary_mask or binary_mask files)\n\n")
            
            for i, folder in enumerate(failed_folders, 1):
                f.write(f"{i}. {folder}\n")
            
            f.write(f"\nTotal failed folders: {len(failed_folders)}\n")
        
        print(f"\nFailed folders report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error saving failed folders report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract parameters from cached files and save as JSON")
    default_root = str(data_root().parent)
    parser.add_argument("--root-path", "-r",
                       default=default_root,
                       help=f"Root path to search for cache folders (default: {default_root})")
    parser.add_argument("--dry-run", "-d", 
                       action="store_true",
                       help="Show what would be done without actually saving files")
    
    args = parser.parse_args()
    
    root_path = Path(args.root_path)
    
    if not root_path.exists():
        print(f"Error: Root path does not exist: {root_path}")
        return 1
    
    print(f"Searching for cache folders in: {root_path}")
    
    # Find all cache folders
    cache_folders = find_cache_folders(root_path)
    
    if not cache_folders:
        print("No cache folders found!")
        return 1
    
    print(f"Found {len(cache_folders)} cache folders:")
    for folder in cache_folders:
        print(f"  {folder}")
    
    # Process each cache folder
    success_count = 0
    total_count = len(cache_folders)
    all_warnings = []
    failed_folders = []  # Track failed folders
    
    for cache_folder in cache_folders:
        parameters, warnings = process_cache_folder(cache_folder)
        all_warnings.extend(warnings)
        
        if parameters:
            if args.dry_run:
                print(f"  [DRY RUN] Would save parameters:")
                print(f"    {json.dumps(parameters, indent=4)}")
            else:
                if save_parameters_json(cache_folder, parameters):
                    success_count += 1
        else:
            print(f"  Skipping {cache_folder} - no valid parameters found")
            failed_folders.append(str(cache_folder))
    
    # Save warnings report if there are any warnings
    if all_warnings and not args.dry_run:
        save_warnings_report(root_path, all_warnings)
    
    # Save failed folders report if there are any failures
    if failed_folders and not args.dry_run:
        save_failed_folders_report(root_path, failed_folders)
    
    print(f"\n" + "="*50)
    print(f"Processing complete:")
    print(f"  Total cache folders: {total_count}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {total_count - success_count}")
    print(f"  Warnings: {len(all_warnings)}")
    
    if args.dry_run:
        print(f"  (This was a dry run - no files were actually saved)")
        if all_warnings:
            print(f"\nWarnings that would be reported:")
            for warning in all_warnings:
                print(f"  - {warning}")
        if failed_folders:
            print(f"\nFolders that would be reported as failed:")
            for folder in failed_folders:
                print(f"  - {folder}")
    else:
        if failed_folders:
            print(f"\nFailed folders (see cache_extraction_failures.txt for details):")
            for folder in failed_folders:
                print(f"  - {folder}")
    
    return 0


if __name__ == "__main__":
    exit(main())
