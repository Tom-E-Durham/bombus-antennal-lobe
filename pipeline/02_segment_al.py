import numpy as np
import sys
from pathlib import Path
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.morphology import ball, closing, dilation
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def has_been_segmented(file_path):
    """
    Check if the file has already been segmented
    """
    file_path = Path(file_path)
    parent_dir = file_path.parent
    filename = file_path.stem
    
    # Check for AL*_seg_counts.csv
    seg_counts_file = parent_dir / f"{filename}_seg_counts.csv"
    if seg_counts_file.exists():
        print(f"Found existing segmentation results: {seg_counts_file}")
        return True
    
    # Check for instance_labels*.tif files in cache
    if filename == "AL1":
        cache_dir = parent_dir / "cache_AL1"
    elif filename == "AL2":
        cache_dir = parent_dir / "cache_AL2"
    else:
        cache_dir = parent_dir / f"cache_{filename}"
    
    if cache_dir.exists():
        instance_files = list(cache_dir.glob("instance_labels*.tif"))
        if instance_files:
            print(f"Found existing instance labels: {len(instance_files)} files")
            return True
    
    return False

def process_file(file_path):
    """
    Process a single file without visualization
    Assumes file has already been filtered by the calling script
    """
    try:
        from bombusal import seg_process
        
        file_path = Path(file_path)
        filename = file_path.stem
        
        print(f"Processing: {file_path}")
        
        # Use automatic parameter loading from parameters.json
        processer = seg_process(data_path=str(file_path), 
                                cache=True, 
                                if_load_cached=True)
        
        # Load the volume
        AL_img = processer.load_volume(str(file_path))
        
        # Pre-process: standardise, enhance contrast, and denoise
        AL_img_denoised = processer.pre_process_volume(AL_img)

        # Create binary mask
        processed_binary_mask = processer.load_processed_binary_mask()
        if processed_binary_mask is not None:
            binary_mask = processed_binary_mask
        else:
            print(f"No processed binary mask found for {file_path}, skipping...")
            return False
        
        # Calculate the volume size (number of voxels) in the binary mask
        volume_size = np.sum(binary_mask)
        print(f"Volume size of binary_mask: {volume_size} voxels")

        # Direct distance transform on binary_mask
        distance = processer.distance_transform(binary_mask)
        distance_original = distance.copy()
        
        # Apply power transformation to enhance centers
        distance_powered = np.power(distance, 2.5)
        print(f"Distance range: {distance.min():.1f} - {distance.max():.1f}")
        print(f"Powered distance range: {distance_powered.min():.1f} - {distance_powered.max():.1f}")

        # Find markers for watershed segmentation
        markers_labeled, distance_enhanced = processer.get_peak_markers(distance_powered, binary_mask)

        # Get elevation map for watershed
        elevation = processer.get_elevation_map(AL_img_denoised, distance_powered)

        # Apply watershed segmentation
        instance_labels_ws = processer.watershed_segmentation(elevation, 
                                                              markers_labeled, 
                                                              binary_mask,
                                                              min_object_size=1000)
        
        # Calculate results
        unique_labels, counts = np.unique(instance_labels_ws, return_counts=True)
        print(f'Number of unique instances found: {len(unique_labels) - 1}')
        
        # Prepare DataFrame with label and count information
        df = pd.DataFrame({'Label': unique_labels, 'Count': counts, 'Ratio': counts / volume_size})

        # Save to CSV with AL-specific naming
        csv_filename = f"{filename}_seg_counts.csv"
        csv_path = file_path.parent / csv_filename
        df.to_csv(csv_path, index=False)
        print(f"Saved label counts to {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_segmentation.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = process_file(file_path)
    sys.exit(0 if success else 1)
