import sys
from pathlib import Path
import logging
import csv
from datetime import datetime
import tifffile
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bombusal.config import data_root, results_dir
from bombusal.io import ConfocalDataset
from bombusal.convex_volume import build_point_cloud, build_convex_hull, mesh_to_shell_volume

LOG_PATH = Path(__file__).with_name("process.log")
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


dataset = ConfocalDataset(str(data_root()))
print("Available samples:", len(dataset))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = results_dir() / f"convex_volumes_{timestamp}.csv"
if not OUTPUT_CSV.exists():
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["species", "id", "al_type", "total_volume"])
        writer.writeheader()

for id in dataset.ids:
    record = dataset[id]
    print(f"Processing: {record['species']}--{id}")
    print(record['cache'])
    for al_type in ["AL1", "AL2"]:
        if not record["cache"].get(al_type):
            continue

        print(f"     {al_type} cache exists, processing.")
        cache_path = Path(record[al_type]).parent / f"cache_{al_type}"
        instance_path = cache_path.rglob("instance_labels_*.tif")
        instance_files = list(instance_path)
        # Check there is only one instance file
        if len(instance_files) != 1:
            logging.warning(
                "Skip id=%s species=%s %s: expected 1 instance file, found %d in %s",
                id,
                record["species"],
                al_type,
                len(instance_files),
                str(cache_path),
            )
            continue
        instance_file = instance_files[0]
        instance_data = tifffile.imread(instance_file)

        # Get information record: xy_unit, z_unit
        info = record.get("info", {})
        al_info = info.get(al_type, {})
        if not al_info:
            logging.warning(
                "Skip id=%s species=%s %s: missing information record",
                id,
                record["species"],
                al_type,
            )
            continue
        xy_unit = al_info.get("xy_unit")
        z_unit = al_info.get("z_unit")
        if xy_unit is None or z_unit is None:
            logging.warning(
                "Skip id=%s species=%s %s: missing units (xy_unit=%s, z_unit=%s)",
                id,
                record["species"],
                al_type,
                str(xy_unit),
                str(z_unit),
            )
            continue

        # Build convex hull and fill holes
        try:
            points_vox = build_point_cloud(instance_data, voxel_size=(1.0, 1.0, 1.0))
            mesh = build_convex_hull(points_vox)
            trimesh.repair.fill_holes(mesh)
            filled_volume = mesh_to_shell_volume(
                mesh,
                output_shape=instance_data.shape,
                pitch=1.0,
                shell_only=False,
                thickness=1,
            )
        except Exception as exc:
            logging.error(
                "Failed processing id=%s species=%s %s: %s",
                id,
                record["species"],
                al_type,
                str(exc),
            )
            continue

        filled_tif_path = cache_path / "convex_filled.tif"
        tifffile.imwrite(str(filled_tif_path), filled_volume.astype(np.uint8))
        print("     Saved:", filled_tif_path)

        # Calculate total volume in real units
        voxel_volume = xy_unit * xy_unit * z_unit
        total_volume = float(filled_volume.sum()) * float(voxel_volume)
        with OUTPUT_CSV.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["species", "id", "al_type", "total_volume"])
            writer.writerow({
                "species": record["species"],
                "id": id,
                "al_type": al_type,
                "total_volume": total_volume,
            })
        print("     Saved volume to:", OUTPUT_CSV)
        print(f"     Total filled volume (real units^3): {total_volume}")
