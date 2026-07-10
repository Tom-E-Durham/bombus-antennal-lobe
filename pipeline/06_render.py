import pyvista as pv
import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import csv
from bombusal.config import data_root, CONFIG, metadata_dir
from bombusal import io as loader

def camera_on_sphere(center, radius, azimuth_deg, elevation_deg, world_up=(0,0,-1)):
    """
    Return a PyVista/VTK camera_position triple (position, focal_point, viewup)
    with the camera on a sphere of given radius around `center`, looking at center.
    azimuth_deg: degrees around Z (0 => +X, 90 => +Y)
    elevation_deg: degrees above XY plane (0 => XY plane, +90 => +Z)
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    # spherical -> cartesian (Z is vertical)
    x = radius * np.cos(el) * np.cos(az)
    y = radius * np.cos(el) * np.sin(az)
    z = radius * np.sin(el)
    pos = np.array(center) + np.array([x, y, z])

    # view direction (from camera to focal point)
    view_dir = np.array(center) - pos
    view_dir /= np.linalg.norm(view_dir)

    # compute a stable viewup vector (avoid near-parallel with view_dir)
    world_up = np.array(world_up, dtype=float)
    if abs(np.dot(view_dir, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])  # fallback up

    right = np.cross(view_dir, world_up)
    right /= np.linalg.norm(right)
    viewup = np.cross(right, view_dir)
    viewup /= np.linalg.norm(viewup)

    return (tuple(pos), tuple(center), tuple(viewup))

def att_lut(n=255, k=10.0, x0=0.3):
    """
    attenuated LUT
    n: must lower than the n_color in add_volume, otherwise the image will not display correctly
    """
    s = np.linspace(0, 1, n)
    a = (1-np.exp(-k*s)) * (1/(1+np.exp(-20*(s-x0))))
    return a / a.max()

def get_rand_angles(num):
    """
    Get random camera angles on a sphere.
    """
    azimuth = np.random.randint(0, 360, size=num)
    elevation = np.random.randint(-90, 90, size=num)
    return azimuth, elevation

def take_photos(file_path, angles):
    reader = pv.get_reader(file_path)
    vol = reader.read()
    scalars = vol.active_scalars_name

    scalar_data = vol.get_array(scalars)
    dimensions = vol.dimensions
    center = tuple(d // 2 for d in dimensions)  
    nx, ny, nz = dimensions
    diag = np.sqrt(nx**2 + ny**2 + nz**2)
    radius = diag * 1.5 # zoom out a little to get full view

    rand_azimuth, rand_elevation = angles
    assert len(rand_azimuth) == len(rand_elevation), "Angle arrays must be the same length"
    photo_num = len(rand_azimuth)

    camera_positions = []
    for i in range(photo_num):
        camera_triple = camera_on_sphere(center, 
                                         radius, 
                                         azimuth_deg=rand_azimuth[i],
                                         elevation_deg=rand_elevation[i])
        camera_positions.append(camera_triple)
    
    vmin, vmax = np.percentile(vol[scalars], (1, 99))
    op = att_lut(255, k=10.0, x0=0.4)

    # Create output directory
    # Create output directory in the same location as the input file
    output_dir = os.path.join(os.path.dirname(file_path), f"fix_renders_{photo_num}_{os.path.basename(file_path).split('.')[0]}")
    os.makedirs(output_dir, exist_ok=True)

    # Create off-screen plotter for batch rendering
    pl = pv.Plotter(off_screen=True, window_size=(800, 800))
    pl.set_background("white")
    
    # Add volume to plotter with your custom opacity function
    actor = pl.add_volume(
        vol, 
        scalars=scalars,
        cmap='gray',
        blending="composite",
        opacity=op,
        clim=[vmin, vmax],
        show_scalar_bar=False
    )
    
    # Iterate through camera positions and take screenshots
    for i, camera_triple in enumerate(camera_positions):
        pl.camera_position = camera_triple
        # Force camera update before screenshot
        pl.render()
        
        # Generate filename with azimuth and elevation info
        azimuth = int(rand_azimuth[i])
        elevation = int(rand_elevation[i])
        filename = f"{output_dir}/view_{i:02d}_az{azimuth:03d}_el{elevation:+03d}.png"
        
        # Take screenshot
        try:
            pl.screenshot(filename)
            
            # Verify the file was actually created
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                print(f"✓ Saved: {filename}")
            else:
                print(f"✗ Failed to save: {filename} (file not created or empty)")
                
        except Exception as e:
            print(f"✗ Error saving {filename}: {str(e)}")
    
    # Close plotter
    pl.close()
    
    # Final verification - count successfully created files
    created_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    success_rate = len(created_files) / photo_num * 100
    print(f"Generated {len(created_files)}/{photo_num} photos ({success_rate:.1f}% success) in directory: {output_dir}")
    
    if len(created_files) < photo_num:
        print(f"⚠️  Warning: {photo_num - len(created_files)} screenshots failed to save")
    
    return camera_positions


def load_render_angles():
    """Camera angles from metadata/render_angles.csv (the same viewpoints are used for every sample).
    Returns (azimuths, elevations) in view order, or None if the file is absent."""
    path = metadata_dir() / "render_angles.csv"
    if not path.exists():
        return None
    azimuth, elevation = [], []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            azimuth.append(int(row["azimuth_deg"]))
            elevation.append(int(row["elevation_deg"]))
    return np.array(azimuth), np.array(elevation)


def main():
    folder_path = str(data_root())
    files = loader.load_dataset(folder_path)

    render_cfg = CONFIG.get("rendering", {})
    photo_num = render_cfg.get("photo_num", 50)          # 50 -> fix_renders_50_<AL>

    angles = load_render_angles()
    if angles is not None:
        # Fixed viewpoints from metadata/render_angles.csv.
        print(f"Using {len(angles[0])} camera angles from metadata/render_angles.csv")
    else:
        # No angle file: draw random viewpoints (seeded for reproducibility).
        seed = render_cfg.get("random_seed")
        if seed is not None:
            np.random.seed(seed)
        angles = get_rand_angles(photo_num)

    for file in files:
        print(f"Processing: {file}")
        take_photos(file, angles)

if __name__ == "__main__":
    # Example usage
    main()