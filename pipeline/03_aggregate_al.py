#!/usr/bin/env python3
"""
Analyze AL segmentation results from CSV files.
This script scans for all AL*_seg_counts.csv files and extracts the data into a summary table.
"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bombusal.config import data_root, results_dir, metadata_dir

def load_unit_information():
    """Load xy_unit and z_unit information from information.csv"""
    info_file = str(metadata_dir() / "information.csv")
    try:
        info_df = pd.read_csv(info_file)
        # Create a lookup dictionary using species, sample, and al_type as key
        unit_lookup = {}
        for _, row in info_df.iterrows():
            key = (row['species'], row['sample'], row['al_type'])
            unit_lookup[key] = {
                'xy_unit': row.get('xy_unit', None),
                'z_unit': row.get('z_unit', None)
            }
        return unit_lookup
    except Exception as e:
        print(f"Warning: Could not load unit information from {info_file}: {e}")
        return {}

def find_seg_counts_files(base_dir):
    """Find all AL*_seg_counts.csv files in the directory tree."""
    seg_files = []
    base_path = Path(base_dir)
    
    for csv_file in base_path.rglob("*_seg_counts.csv"):
        filename = csv_file.name
        # Check if it's an AL file (AL1 or AL2)
        if "AL1_seg_counts.csv" in filename or "AL2_seg_counts.csv" in filename:
            seg_files.append(csv_file)
    
    return sorted(seg_files)

def analyze_segmentation_file(csv_file, unit_lookup):
    """Analyze a single segmentation CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get file info
        file_path = Path(csv_file)
        relative_path = file_path.relative_to(data_root())
        species = relative_path.parts[0]
        sample = relative_path.parts[1]
        
        # Determine AL type from filename
        if "AL1_seg_counts.csv" in file_path.name:
            al_type = "AL1"
        elif "AL2_seg_counts.csv" in file_path.name:
            al_type = "AL2"
        else:
            al_type = "Unknown"
        
        # Get unit information
        key = (species, sample, al_type)
        units = unit_lookup.get(key, {'xy_unit': None, 'z_unit': None})
        xy_unit = units['xy_unit']
        z_unit = units['z_unit']
        
        # Analyze the data
        if 'Label' in df.columns and 'Count' in df.columns:
            # Remove background (label 0 if present)
            df_filtered = df[df['Label'] != 0]
            
            total_instances = len(df_filtered)
            total_voxels = df_filtered['Count'].sum()
            
            # Calculate real volume if units are available
            real_volume_um3 = None
            if xy_unit is not None and z_unit is not None:
                # Volume per voxel = xy_unit^2 * z_unit (in μm³)
                voxel_volume_um3 = (xy_unit ** 2) * z_unit
                real_volume_um3 = total_voxels * voxel_volume_um3
            
            return {
                'species': species,
                'sample': sample,
                'al_type': al_type,
                'total_instances': total_instances,
                'total_voxels': total_voxels,
                'real_volume_um3': real_volume_um3
            }
        else:
            print(f"Warning: Expected columns not found in {csv_file}")
            return None
            
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None

def main():
    """Main analysis function."""
    # Configuration
    base_dir = str(data_root())
    output_dir = str(results_dir())
    
    print("Loading unit information from information.csv...")
    unit_lookup = load_unit_information()
    print(f"Loaded unit information for {len(unit_lookup)} entries")
    
    print("Scanning for AL*_seg_counts.csv files...")
    seg_files = find_seg_counts_files(base_dir)
    
    print(f"Found {len(seg_files)} segmentation CSV files")
    
    if not seg_files:
        print("No segmentation CSV files found!")
        return
    
    # Analyze each file
    results = []
    print("\nAnalyzing files...")
    for i, csv_file in enumerate(seg_files, 1):
        print(f"  [{i}/{len(seg_files)}] {csv_file.name}")
        result = analyze_segmentation_file(csv_file, unit_lookup)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results to analyze!")
        return
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results to CSV
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_file = output_path / "segmentation_analysis_results.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Print basic summary
    print(f"\n=== BASIC SUMMARY ===")
    print(f"Total files processed: {len(df_results)}")
    print(f"Total instances found: {df_results['total_instances'].sum():,}")
    print(f"Average instances per file: {df_results['total_instances'].mean():.1f}")
    
    # Print volume summary if available
    valid_volumes = df_results['real_volume_um3'].dropna()
    if len(valid_volumes) > 0:
        print(f"\n=== VOLUME SUMMARY ===")
        print(f"Files with volume data: {len(valid_volumes)}")
        print(f"Total volume: {valid_volumes.sum():.2f} μm³")
        print(f"Average volume per file: {valid_volumes.mean():.2f} μm³")
        print(f"Volume range: {valid_volumes.min():.2f} - {valid_volumes.max():.2f} μm³")
    else:
        print(f"\nWarning: No volume data could be calculated (missing unit information)")
    
    print(f"\nData extraction complete! Results saved to: {output_dir}")
    print("Run 'python segmentation_summary.py' for detailed analysis and visualizations.")

if __name__ == "__main__":
    main()
