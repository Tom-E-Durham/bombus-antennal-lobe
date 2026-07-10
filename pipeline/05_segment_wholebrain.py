import os
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import napari
from pathlib import Path
from collections import defaultdict
from skimage import exposure, measure, feature
from skimage.filters import threshold_otsu, gaussian, sobel
from scipy.ndimage import distance_transform_edt, binary_fill_holes # For distance transform and hole filling
from skimage.segmentation import watershed # Added watershed
from skimage.morphology import cube, closing, binary_closing, binary_opening
import csv
import gc

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bombusal.config import data_root, CONFIG

class seg_process():
    def __init__(self, 
                 data_path, 
                 cache = False,
                 global_mean = 5685.8,
                 global_std = 8127.3,
                 lower_percentile = 80,
                 upper_percentile = 99,
                 gaussian_sigma = 0.5,
                 threshold_factor = 0.5,
                 peak_min_distance = 35,
                 cc_min_volume = 1000,
                 closing_kernel_size = 3,
                 opening_kernel_size = 3,
                 fill_holes_iterations = 1):
        
        self.data_path = Path(data_path)
        self.cache = cache
        if cache:
            self.cache_path = self.data_path.parent / 'cache_wb'
            self.cache_path.mkdir(parents=True, exist_ok=True)
        # standardisation parameters
        self.global_mean = global_mean
        self.global_std = global_std
        # contract enhancement parameters
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        # denoising parameters
        self.gaussian_sigma = gaussian_sigma
        # binary segmentation parameters
        self.threshold_factor = threshold_factor
        # peak detection parameters
        self.peak_min_distance = peak_min_distance
        # connected components parameters
        self.cc_min_volume = cc_min_volume
        # morphological operations parameters (cube kernel size)
        self.closing_kernel_size = closing_kernel_size
        self.opening_kernel_size = opening_kernel_size
        self.fill_holes_iterations = fill_holes_iterations
    
    def cache_results(self, data, filename):
        if not self.cache:
            return None
        
        cache_file = self.cache_path / f"{filename}.tif"
        if np.issubdtype(data.dtype, np.integer):
            tifffile.imwrite(str(cache_file), data.astype(np.int32))
        else:
            tifffile.imwrite(str(cache_file), data.astype(np.float32))
        print(f"Cached processed volume at {cache_file}")
    
    def load_cached(self, filename):
        if not self.cache:
            return None
        
        cache_file = self.cache_path / f"{filename}.tif"
        if cache_file.exists():
            data = tifffile.imread(str(cache_file))
            print(f"Loaded cached volume from {cache_file}")
            return data
        else:
            print(f"No cached file found at {cache_file}")
            return None
    
    def load_volume(self, file_path):
        """
        Load a TIFF volume from the specified file path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")
        
        volume = tifffile.imread(file_path)
        print(f'Loaded file: {file_path}')
        print(f'Shape: {volume.shape}')
        print(f'Data type: {volume.dtype}')
        
        volume_standardised = self.zscore_clip_and_rescale(volume)
        
        return volume_standardised

    def zscore_clip_and_rescale(self, volume):
        """
        Standardise the volume using z-score normalization and rescale to 0-1.
        """
        
        standardised = (volume - self.global_mean) / self.global_std
        print(f"Standardised volume range: {standardised.min()} to {standardised.max()}")
        print(f"Standardised volume: mean={standardised.mean()}, std={standardised.std()}")
        standardised = ((standardised - standardised.min()) / 
                        (standardised.max() - standardised.min()))
        print(f"Rescaled volume range: {standardised.min()} to {standardised.max()}")
        print(f"Rescaled volume: mean={standardised.mean()}, std={standardised.std()}")

        return standardised

    def contrast_enhancement(self, volume):
        """
        Enhance contrast of the 3D volume using specified percentiles.
        """
        filename = f'contrast_enhanced_C{self.lower_percentile}_{self.upper_percentile}'
        volume_rescaled = self.load_cached(filename)
        if volume_rescaled is not None:
            return volume_rescaled
        
        p_low, p_high = np.percentile(volume, 
                                      (self.lower_percentile, 
                                       self.upper_percentile))
        volume_rescaled = exposure.rescale_intensity(volume, 
                                                    in_range=(p_low, p_high), 
                                                    out_range=(0, 1))
        # Calculate the actual intensity values at these percentiles for the current volume
        print(f"Auto-contrast: using intensity range [{p_low:.2f} - {p_high:.2f}] (based on {self.lower_percentile}th and {self.upper_percentile}th percentiles).")

        if self.cache:
            self.cache_results(volume_rescaled, filename)
        
        return volume_rescaled
    
    def denoise_volume(self, volume):
        """
        Apply Gaussian filter to denoise the 3D volume.
        """
        filename = f'denoised_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}'
        denoised = self.load_cached(filename)
        if denoised is not None:
            return denoised
        
        denoised = gaussian(volume, 
                            sigma=self.gaussian_sigma, 
                            preserve_range=True, 
                            mode='reflect')
                
        if self.cache:
            self.cache_results(denoised, filename)
        
        return denoised
    
    def pre_process_volume(self, volume_standardised):
        """
        Pre-process the volume by standardizing, enhancing contrast, and denoising.
        """
        filename = f'pre_processed_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}'
        pre_processed = self.load_cached(filename)
        if pre_processed is not None:
            return pre_processed
        
        if self.cache:
            self.cache = False
            volume_contrast_enhanced = self.contrast_enhancement(volume_standardised)
            volume_denoised = self.denoise_volume(volume_contrast_enhanced)
            self.cache = True
            self.cache_results(volume_denoised, filename)
        else:
            volume_contrast_enhanced = self.contrast_enhancement(volume_standardised)
            volume_denoised = self.denoise_volume(volume_contrast_enhanced)
            
        return volume_denoised
    
    def binary_segmentation(self, volume):
        """
        Create a binary mask from the volume using Otsu's thresholding.
        """
        filename = f'binary_mask_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}_B{self.threshold_factor}'
        binary_mask = self.load_cached(filename)
        if binary_mask is not None:
            return binary_mask
        
        thresh = threshold_otsu(volume)
        binary_mask = volume > (thresh * self.threshold_factor)
        binary_mask = closing(binary_mask, cube(3))

        if self.cache:
            self.cache_results(binary_mask, filename)
        return binary_mask
    
    def fill_and_smooth_mask(self, binary_mask):
        """
        Fill holes and smooth the binary mask using morphological operations.
        1. Morphological closing: bridge gaps and fill small holes
        2. Binary hole filling: fill all enclosed holes
        3. Morphological opening: remove small protrusions
        """
        filename = f'morphology_processed_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}_B{self.threshold_factor}_CL{self.closing_kernel_size}_OP{self.opening_kernel_size}_FH{self.fill_holes_iterations}'
        processed_mask = self.load_cached(filename)
        if processed_mask is not None:
            return processed_mask
        
        volume_original = int(np.sum(binary_mask))
        
        # Step 1: Morphological closing - bridge gaps and connect nearby regions
        closing_element = cube(self.closing_kernel_size)
        print(f"Applying morphological closing with cube kernel {self.closing_kernel_size}x{self.closing_kernel_size}x{self.closing_kernel_size}...")
        closed_mask = binary_closing(binary_mask, closing_element)
        volume_after_closing = int(np.sum(closed_mask))
        print(f"  After closing: {volume_after_closing - volume_original:+d} voxels")
        
        # Step 2: Fill all enclosed holes (with iterations for stubborn holes)
        print(f"Filling enclosed holes (iterations: {self.fill_holes_iterations})...")
        filled_mask = closed_mask
        for i in range(self.fill_holes_iterations):
            filled_mask = binary_fill_holes(filled_mask)
            if i == 0:
                volume_after_first_fill = int(np.sum(filled_mask))
                voxels_filled = volume_after_first_fill - volume_after_closing
                print(f"  After hole filling iteration 1: {voxels_filled:+d} voxels")
            elif i > 0:
                volume_after_iter = int(np.sum(filled_mask))
                voxels_filled_iter = volume_after_iter - volume_after_first_fill
                if voxels_filled_iter > 0:
                    print(f"  After hole filling iteration {i+1}: {voxels_filled_iter:+d} additional voxels")
        
        volume_after_filling = int(np.sum(filled_mask))
        total_filled = volume_after_filling - volume_after_closing
        if self.fill_holes_iterations > 1:
            print(f"  Total filled by all iterations: {total_filled:+d} voxels")
        
        # Step 3: Morphological opening - remove small protrusions and smooth
        opening_element = cube(self.opening_kernel_size)
        print(f"Applying morphological opening with cube kernel {self.opening_kernel_size}x{self.opening_kernel_size}x{self.opening_kernel_size}...")
        processed_mask = binary_opening(filled_mask, opening_element)
        volume_final = int(np.sum(processed_mask))
        print(f"  After opening: {volume_final - volume_after_filling:+d} voxels")
        
        print(f"Total volume change: {volume_original} -> {volume_final} ({volume_final - volume_original:+d} voxels)")
        
        if self.cache:
            self.cache_results(processed_mask, filename)
        
        return processed_mask
    
    def filter_connected_components(self, binary_mask):
        """
        Filter binary mask using 3D connected components analysis.
        Remove small connected components below the volume threshold to reduce noise.
        """
        filename = f'cc_filtered_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}_B{self.threshold_factor}_CL{self.closing_kernel_size}_OP{self.opening_kernel_size}_FH{self.fill_holes_iterations}_CC{self.cc_min_volume}'
        filtered_mask = self.load_cached(filename)
        if filtered_mask is not None:
            return filtered_mask
        
        # Perform 3D connected components labeling
        labeled_mask = measure.label(binary_mask, connectivity=3)
        print(f"Total connected components found: {labeled_mask.max()}")
        
        # Get properties of each connected component
        regions = measure.regionprops(labeled_mask)
        
        # Filter components by volume
        filtered_mask = np.zeros_like(binary_mask, dtype=bool)
        kept_components = 0
        removed_components = 0
        
        for region in regions:
            component_volume = region.area  # area is the number of voxels (volume in 3D)
            if component_volume >= self.cc_min_volume:
                # Keep this component
                filtered_mask[labeled_mask == region.label] = True
                kept_components += 1
            else:
                removed_components += 1
        
        print(f"Connected components kept: {kept_components}")
        print(f"Connected components removed (noise): {removed_components}")
        print(f"Volume threshold: {self.cc_min_volume} voxels")
        
        if self.cache:
            self.cache_results(filtered_mask, filename)
        
        return filtered_mask
    
if __name__ == "__main__":

    confocal_dir = str(data_root())
    wholebrain_paths = list(Path(confocal_dir).rglob("wholebrain.tif"))
    print(f"Found {len(wholebrain_paths)} wholebrain files:")

    # Whole-brain parameters (see config.yaml).
    wb = CONFIG["wholebrain"]
    global_mean = wb["global_mean"]
    global_std = wb["global_std"]
    lower_percentile = wb["lower_percentile"]   # 0  -> full-range contrast stretch (no-op)
    upper_percentile = wb["upper_percentile"]   # 100
    gaussian_sigma = wb["gaussian_sigma"]
    threshold_factor = wb["threshold_factor"]   # Otsu * 0.2
    cc_min_volume = wb["cc_min_volume"]
    closing_kernel_size = wb["closing_kernel_size"]   # larger kernel to close the ring/donut gaps
    opening_kernel_size = wb["opening_kernel_size"]   # keep small to avoid undoing the hole filling
    fill_holes_iterations = wb["fill_holes_iterations"]

    # Read existing data from CSV
    csv_path = Path(confocal_dir) / 'wholebrain_volumes_with_real_volume.csv'
    processed_data = {}
    xy_z_units = {}

    # Read xy_unit and z_unit from original CSV if available
    original_csv_path = Path(confocal_dir) / 'wholebrain_volumes.csv'
    if original_csv_path.exists():
        with open(original_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                b_id = row['B_id']
                if b_id not in xy_z_units and row.get('xy_unit') and row.get('z_unit'):
                    xy_z_units[b_id] = {
                        'xy_unit': float(row['xy_unit']),
                        'z_unit': float(row['z_unit'])
                    }

    # Skip already processed with valid real_volume
    processed_bids = set()
    for b_id, data in processed_data.items():
        if data.get('real_volume_um3') and data['real_volume_um3'].strip():
            processed_bids.add(b_id)
            print(f"Already processed: {b_id}")

    # Skip excluded samples (e.g. B471: whole-brain volume too large / artefact).
    for excluded in wb.get("exclude_samples", []):
        processed_bids.add(excluded)

    # Create a single processer instance with a dummy path (must not be None)
    dummy_path = wholebrain_paths[0] if wholebrain_paths else '.'
    processer = seg_process(
        data_path=dummy_path,
        cache=True,
        global_mean=global_mean,
        global_std=global_std,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        gaussian_sigma=gaussian_sigma,
        threshold_factor=threshold_factor,
        cc_min_volume=cc_min_volume,
        closing_kernel_size=closing_kernel_size,
        opening_kernel_size=opening_kernel_size,
        fill_holes_iterations=fill_holes_iterations
    )

    for path in wholebrain_paths:
        # Use the last part of the path as B_id
        b_id = path.parts[-2]
        
        if b_id in processed_bids:
            print(f"\n{'='*80}")
            print(f"Skipping {b_id}, already processed.")
            print(f"{'='*80}\n")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {b_id}...")
        print(f"{'='*80}\n")

        try:
            processer.data_path = path
            processer.cache_path = path.parent / 'cache_wb'
            processer.cache_path.mkdir(parents=True, exist_ok=True)

            # Load the volume
            volume = processer.load_volume(str(path))

            # Pre-process: standardise, enhance contrast, and denoise
            volume_denoised = processer.pre_process_volume(volume)

            # Create binary mask
            binary_mask = processer.binary_segmentation(volume_denoised).astype(np.uint8)

            # Apply morphological operations: closing, fill holes, opening
            binary_mask_smooth = processer.fill_and_smooth_mask(binary_mask.astype(bool))

            # Apply connected components filtering to remove noise
            binary_mask_filtered = processer.filter_connected_components(binary_mask_smooth)

            # Calculate the volume size
            volume_size = int(np.sum(binary_mask_filtered))
            
            print(f"\n{'='*80}")
            print(f"Final volume size for {b_id}: {volume_size} voxels")
            
            # Calculate real volume if we have the units
            real_volume_um3 = None
            xy_unit = None
            z_unit = None
            
            if b_id in xy_z_units:
                xy_unit = xy_z_units[b_id]['xy_unit']
                z_unit = xy_z_units[b_id]['z_unit']
                voxel_volume_um3 = (xy_unit ** 2) * z_unit
                real_volume_um3 = volume_size * voxel_volume_um3
                print(f"xy_unit: {xy_unit}, z_unit: {z_unit}")
                print(f"Real volume: {real_volume_um3:.2f} μm³")
            else:
                print(f"Warning: No xy_unit/z_unit found for {b_id}")
            
            print(f"{'='*80}\n")

            # Immediately write to CSV (real-time update)
            # Read all existing data
            all_data = {}
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_data[row['B_id']] = row
            
            # Update with new result
            all_data[b_id] = {
                'B_id': b_id,
                'volume_size': volume_size,
                'xy_unit': xy_unit if xy_unit else '',
                'z_unit': z_unit if z_unit else '',
                'real_volume_um3': real_volume_um3 if real_volume_um3 else ''
            }
            
            # Write back to CSV immediately
            with open(csv_path, 'w', newline='') as f:
                fieldnames = ['B_id', 'volume_size', 'xy_unit', 'z_unit', 'real_volume_um3']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for bid in sorted(all_data.keys()):
                    writer.writerow(all_data[bid])
            
            print(f"✓ Saved {b_id} to {csv_path}\n")

        except Exception as e:
            print(f"Error processing {b_id}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up memory
            try:
                del volume, volume_denoised, binary_mask, binary_mask_smooth, binary_mask_filtered
            except:
                pass
            gc.collect()
