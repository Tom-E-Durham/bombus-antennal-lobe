import os
import re
import numpy as np
import tifffile
from pathlib import Path
from skimage import exposure, measure, feature
from skimage.filters import threshold_otsu, gaussian, sobel
from scipy.ndimage import distance_transform_edt # For distance transform
from skimage.segmentation import watershed # Added watershed
from skimage.morphology import ball, closing
import pandas as pd

class seg_process():
    def __init__(self, 
                 data_path, 
                 cache = False,
                 if_load_cached = False,
                 global_mean = 5685.8,
                 global_std = 8127.3,
                 lower_percentile = 80,
                 upper_percentile = 99,
                 gaussian_sigma = 0.5,
                 threshold_factor = 0.5,
                 peak_min_distance = 20,
                 distance_threshold = 60):
        
        self.data_path = Path(data_path)
        self.cache = cache
        self.if_load_cached = if_load_cached
        
        if cache:
            if self.data_path.name == 'AL1.tif':
                self.cache_path = self.data_path.parent / 'cache_AL1'
            elif self.data_path.name == 'AL2.tif':
                self.cache_path = self.data_path.parent / 'cache_AL2'
            elif self.data_path.name == 'wholebrain.tif':
                self.cache_path = self.data_path.parent / 'cache_wb'
            else:
                # Try to infer cache folder name from file name
                file_stem = self.data_path.stem.lower()
                if 'al1' in file_stem:
                    self.cache_path = self.data_path.parent / 'cache_AL1'
                elif 'al2' in file_stem:
                    self.cache_path = self.data_path.parent / 'cache_AL2'
                elif 'wb' in file_stem or 'wholebrain' in file_stem:
                    self.cache_path = self.data_path.parent / 'cache_wb'
                else:
                    # Fallback to generic cache folder
                    self.cache_path = self.data_path.parent / f'cache_{self.data_path.stem}'
                    print(f"Warning: Using inferred cache path: {self.cache_path}")
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
        # distance transform threshold
        self.distance_threshold = distance_threshold
        
        if self.if_load_cached:
            print("Loading cached parameters...")
            
            # First try to load from parameters.json file
            json_file = self.cache_path / "parameters.json"
            if json_file.exists():
                try:
                    import json
                    with open(json_file, 'r', encoding='utf-8') as f:
                        cached_params = json.load(f)
                    
                    # Update parameters from JSON
                    if 'global_mean' in cached_params and cached_params['global_mean'] is not None:
                        self.global_mean = cached_params['global_mean']
                    if 'global_std' in cached_params and cached_params['global_std'] is not None:
                        self.global_std = cached_params['global_std']
                    if 'lower_percentile' in cached_params and cached_params['lower_percentile'] is not None:
                        self.lower_percentile = cached_params['lower_percentile']
                    if 'upper_percentile' in cached_params and cached_params['upper_percentile'] is not None:
                        self.upper_percentile = cached_params['upper_percentile']
                    if 'gaussian_sigma' in cached_params and cached_params['gaussian_sigma'] is not None:
                        self.gaussian_sigma = cached_params['gaussian_sigma']
                    if 'threshold_factor' in cached_params and cached_params['threshold_factor'] is not None:
                        self.threshold_factor = cached_params['threshold_factor']
                    if 'peak_min_distance' in cached_params and cached_params['peak_min_distance'] is not None:
                        self.peak_min_distance = cached_params['peak_min_distance']
                    if 'distance_threshold' in cached_params and cached_params['distance_threshold'] is not None:
                        self.distance_threshold = cached_params['distance_threshold']
                    
                    print(f"Loaded parameters from {json_file}")
                    print(f"Parameters: global_mean={self.global_mean}, global_std={self.global_std}")
                    print(f"            lower_percentile={self.lower_percentile}, upper_percentile={self.upper_percentile}")
                    print(f"            gaussian_sigma={self.gaussian_sigma}, threshold_factor={self.threshold_factor}")
                    print(f"            peak_min_distance={self.peak_min_distance}, distance_threshold={self.distance_threshold}")
                    
                except Exception as e:
                    print(f"Error loading parameters from {json_file}: {e}")
                    print("Falling back to filename parsing...")
                    self._load_from_filenames()
            else:
                print("No parameters.json found, trying to parse from filenames...")
                self._load_from_filenames()
    
    def _load_from_filenames(self):
        """
        Fallback method to load parameters from cached filenames.
        """
        # Search for a processed_binary_mask file in the cache folder by pattern
        mask_files = list(self.cache_path.glob("processed_binary_mask_C*-*_D*_B*.tif"))
        if mask_files:
            mask_file = mask_files[0]
            print(f"Found cached processed_binary_mask: {mask_file.name}")
            # Extract parameters from filename using regex
            match = re.match(r"processed_binary_mask_C(\d+)-(\d+)_D([0-9.]+)_B([0-9.]+)\.tif", mask_file.name)
            if match:
                self.lower_percentile = int(match.group(1))
                self.upper_percentile = int(match.group(2))
                self.gaussian_sigma = float(match.group(3))
                self.threshold_factor = float(match.group(4))
                print(f"Loaded parameters from cached mask: lower_percentile={self.lower_percentile}, upper_percentile={self.upper_percentile}, gaussian_sigma={self.gaussian_sigma}, threshold_factor={self.threshold_factor}")
                
                # Try to get peak_min_distance and distance_threshold from peak_markers file
                marker_files = list(self.cache_path.glob("peak_markers_C*-*_D*_B*_PM*.tif"))
                marker_files.extend(list(self.cache_path.glob("peak_markers_C*-*_D*_B*_DT*_PM*.tif")))
                if marker_files:
                    marker_file = marker_files[0]
                    print(f"Found cached peak_markers: {marker_file.name}")
                    
                    # Try new format first (with DT parameter)
                    dt_match = re.search(r"DT([0-9]+)_PM([0-9]+)\.tif", marker_file.name)
                    if dt_match:
                        self.distance_threshold = int(dt_match.group(1))
                        self.peak_min_distance = int(dt_match.group(2))
                        print(f"Loaded distance_threshold from markers: {self.distance_threshold}")
                        print(f"Loaded peak_min_distance from markers: {self.peak_min_distance}")
                    else:
                        # Fallback to old format (without DT parameter)
                        marker_match = re.search(r"PM([0-9]+)\.tif", marker_file.name)
                        if marker_match:
                            self.peak_min_distance = int(marker_match.group(1))
                            print(f"Loaded peak_min_distance from markers: {self.peak_min_distance}")
                            print(f"Using default distance_threshold: {self.distance_threshold}")
            else:
                print("Could not parse parameters from cached mask filename.")
        else:
            print("No processed_binary_mask file found in cache.")
    
    def cache_results(self, data, filename):
        cache_file = self.cache_path / f"{filename}.tif"
        if np.issubdtype(data.dtype, np.integer):
            tifffile.imwrite(str(cache_file), data.astype(np.int32))
        else:
            tifffile.imwrite(str(cache_file), data.astype(np.float32))
        print(f"Cached processed volume at {cache_file}")
    
    def load_cached(self, filename):
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
        
        return volume

    def zscore_clip_and_rescale(self, volume):
        """
        Standardise the volume using z-score normalization and rescale to 0-1.
        """
        
        standardised = self.load_cached('standardised')
        if standardised is not None:
            return standardised
        
        standardised = (volume - self.global_mean) / self.global_std
        print(f"Standardised volume range: {standardised.min()} to {standardised.max()}")
        print(f"Standardised volume: mean={standardised.mean()}, std={standardised.std()}")
        standardised = ((standardised - standardised.min()) / 
                        (standardised.max() - standardised.min()))
        print(f"Rescaled volume range: {standardised.min()} to {standardised.max()}")
        print(f"Rescaled volume: mean={standardised.mean()}, std={standardised.std()}")
        
        if self.cache:
            filename = 'standardised'
            self.cache_results(standardised, filename)
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
    
    def pre_process_volume(self, volume):
        """
        Pre-process the volume by standardizing, enhancing contrast, and denoising.
        """
        filename = f'pre_processed_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}'
        pre_processed = self.load_cached(filename)
        if pre_processed is not None:
            return pre_processed
        
        volume_standardised = self.zscore_clip_and_rescale(volume)
        volume_contrast_enhanced = self.contrast_enhancement(volume_standardised)
        volume_denoised = self.denoise_volume(volume_contrast_enhanced)
        
        if self.cache:
            self.cache_results(volume_denoised, filename)

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
        binary_mask = closing(binary_mask, ball(2))

        if self.cache:
            self.cache_results(binary_mask, filename)
        return binary_mask
    
    def load_processed_binary_mask(self):
        """
        Load the binary mask from the cache or compute it if not cached.
        """
        filename = f'processed_binary_mask_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}_B{self.threshold_factor}'
        if os.path.exists(self.cache_path / f"{filename}.tif"):
            return self.load_cached(filename)
        else:
            return None
    
    def get_strong_binary_mask(self, volume, binary_mask):
        """
        Create a strong binary mask based on the binary mask and volume intensities.
        """
        filename = f'strong_binary_mask_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}_B{self.threshold_factor}_DT{self.distance_threshold}'
        strong_binary_mask = self.load_cached(filename)
        if strong_binary_mask is not None:
            return strong_binary_mask
        
        masked_intensities = volume[binary_mask > 0]
        threshold = np.percentile(masked_intensities, self.distance_threshold)
        print(f"Calculated threshold ({self.distance_threshold}th percentile): {threshold:.3f}")
        strong_binary_mask = (volume > threshold) & binary_mask

        strong_binary_mask = closing(strong_binary_mask, ball(3))

        if self.cache:
            self.cache_results(strong_binary_mask, filename)
        return strong_binary_mask
    
    def distance_transform(self, binary_mask=None):
        """
        Compute the distance transform of the binary mask.
        """
        filename = f'distance_transform_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}_B{self.threshold_factor}_DT{self.distance_threshold}'
        distance = self.load_cached(filename)
        if distance is not None:
            return distance
        
        # Compute the distance transform
        distance = distance_transform_edt(binary_mask)
        if self.cache:
            self.cache_results(distance, filename)
        
        return distance

    def get_elevation_map(self, volume, distance, alpha=0.6):
        """
        Compute the elevation map from the volume and distance transform.
        """

        # Compute elevation map for watershed
        print("Computing elevation map for watershed...")
        # Normalize both components
        distance_norm = distance / (distance.max() + 1e-8)
        gradient = sobel(volume.astype(np.float32))
        gradient_norm = gradient / (gradient.max() + 1e-8)

        # alpha = 0.6  # Moderate emphasis on distance (preserve blob coverage)
        elevation = gradient_norm * (1 - alpha) + (1 - distance_norm) * alpha
        elevation = elevation ** 1.3  # Optional power enhancement for contrast
        print(f"Distance weight (alpha): {alpha}, Gradient weight: {1-alpha}")

        return elevation

    def get_peak_markers(self, distance, binary_mask):
        """
        Find markers for watershed segmentation using local maxima in the distance transform.
        """
        filename = f"peak_markers_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}_B{self.threshold_factor}_DT{self.distance_threshold}_PM{self.peak_min_distance}"
        markers_labeled = self.load_cached(filename)
        
        
        # Multi-level threshold strategy: preserve center regions of different intensities
        # Method 1: Percentile-based thresholds
        distance_threshold_high = np.percentile(distance[distance > 0], 80)  # Strong centers
        distance_threshold_med = np.percentile(distance[distance > 0], 60)   # Medium centers
        distance_threshold_low = np.percentile(distance[distance > 0], 40)   # Weak centers

        print(f"Distance thresholds - High: {distance_threshold_high:.1f}, "
            f"Med: {distance_threshold_med:.1f}, Low: {distance_threshold_low:.1f}")

        # Create filtered distance map
        distance_filtered = distance.copy()
        distance_filtered[distance_filtered < distance_threshold_med] = 0  # Use medium threshold

        # Valley suppression to enhance blob separation
        valley_threshold = np.percentile(distance_filtered[distance_filtered > 0], 25)
        valley_mask = (distance_filtered > 0) & (distance_filtered < valley_threshold)
        distance_filtered[valley_mask] *= 0.4  # Suppress connection regions
        
        if markers_labeled is not None:
            return markers_labeled, distance_filtered
        
        markers_coords = feature.peak_local_max(distance_filtered,
                                                min_distance=self.peak_min_distance,
                                                threshold_rel=0,
                                                footprint=ball(2),
                                                labels=binary_mask)
        markers_mask = np.zeros(distance.shape, dtype=bool)
        markers_mask[tuple(markers_coords.T)] = True
        markers_labeled = measure.label(markers_mask)
        
        print(f"Number of markers found: {markers_coords.shape[0]}")
        
        if self.cache:
            self.cache_results(markers_labeled, filename)

        return markers_labeled, distance_filtered

    def watershed_segmentation(self, 
                               elevation, 
                               markers_labeled, 
                               binary_mask, 
                               min_object_size=1000,
                               save_result=True):
        """
        Apply watershed segmentation to the elevation map using the provided markers.
        """
        file_name = f'instance_labels_C{self.lower_percentile}-{self.upper_percentile}_D{self.gaussian_sigma}_B{self.threshold_factor}_DT{self.distance_threshold}_PM{self.peak_min_distance}'
        instance_labels_ws = self.load_cached(file_name)
        if instance_labels_ws is not None:
            return instance_labels_ws
        
        instance_labels_ws = watershed(elevation, markers_labeled, mask=binary_mask)
        
        print(f'Number of unique instances before removing small labels: {len(np.unique(instance_labels_ws)) - 1}')
        if min_object_size is not None:
            # Remove small objects from the watershed output
            unique_labels, counts = np.unique(instance_labels_ws, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if label == 0: continue
                if count < min_object_size:
                    instance_labels_ws[instance_labels_ws == label] = 0
        print(f'Watershed instance labels shape: {instance_labels_ws.shape}')
        print(f'Number of unique instances found (watershed, excluding background): {len(np.unique(instance_labels_ws)) - 1}')
        
        unique_labels, counts = np.unique(instance_labels_ws, return_counts=True)
        # Prepare DataFrame with label and count information
        volume_size = np.sum(binary_mask)
        df = pd.DataFrame({'Label': unique_labels, 'Count': counts, 'Ratio': counts / volume_size})

        # Save to CSV in the same directory as the input path
        csv_path = Path(self.data_path).parent / "instance_label_counts.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved label counts to {csv_path}")
        if self.cache:
            self.cache_results(instance_labels_ws, file_name)
        
        return instance_labels_ws