import sys
from pathlib import Path

import tifffile
import numpy as np
import trimesh


def build_point_cloud(instance_data, voxel_size=(1.0, 1.0, 1.0)):
    coords_zyx = np.argwhere(instance_data > 0)
    if coords_zyx.size == 0:
        raise ValueError("No foreground voxels found in instance_data.")
    coords_xyz = coords_zyx[:, [2, 1, 0]]
    scale = np.array(voxel_size, dtype=np.float32)
    return coords_xyz.astype(np.float32) * scale


def build_convex_hull(points_xyz):
    if points_xyz.shape[0] < 4:
        raise ValueError("Need at least 4 non-coplanar points for a 3D convex hull.")
    cloud = trimesh.points.PointCloud(points_xyz)
    return cloud.convex_hull


def mesh_to_shell_volume(mesh, output_shape, pitch=1.0, shell_only=True, thickness=1):
    voxel_grid = mesh.voxelized(pitch)
    filled = voxel_grid.fill()
    if filled is not None:
        voxel_grid = filled

    matrix_xyz = voxel_grid.matrix.astype(np.uint8)
    origin_xyz = getattr(voxel_grid, "origin", None)
    if origin_xyz is None:
        origin_xyz = voxel_grid.bounds[0]

    matrix_zyx = np.transpose(matrix_xyz, (2, 1, 0))

    if shell_only:
        shell = matrix_zyx.copy()
        for _ in range(max(1, thickness)):
            eroded = shell.copy()
            for axis in range(3):
                eroded &= np.roll(shell, 1, axis=axis)
                eroded &= np.roll(shell, -1, axis=axis)
            shell = shell & (~eroded)
        matrix_zyx = shell

    offset_xyz = np.rint(origin_xyz / pitch).astype(np.int64)
    offset_zyx = (offset_xyz[2], offset_xyz[1], offset_xyz[0])

    volume = np.zeros(output_shape, dtype=np.uint8)
    z0, y0, x0 = offset_zyx
    z1 = z0 + matrix_zyx.shape[0]
    y1 = y0 + matrix_zyx.shape[1]
    x1 = x0 + matrix_zyx.shape[2]

    z0c = max(z0, 0)
    y0c = max(y0, 0)
    x0c = max(x0, 0)
    z1c = min(z1, output_shape[0])
    y1c = min(y1, output_shape[1])
    x1c = min(x1, output_shape[2])

    if z0c < z1c and y0c < y1c and x0c < x1c:
        mz0 = z0c - z0
        my0 = y0c - y0
        mx0 = x0c - x0
        mz1 = mz0 + (z1c - z0c)
        my1 = my0 + (y1c - y0c)
        mx1 = mx0 + (x1c - x0c)
        volume[z0c:z1c, y0c:y1c, x0c:x1c] = matrix_zyx[mz0:mz1, my0:my1, mx0:mx1]

    return volume


def mask_to_uint8(mask):
    return mask.astype(np.uint8)


if __name__ == "__main__":
    # Demo: read an instance-labels tif path from the command line and write the
    # convex-hull filled/shell volumes next to it.
    file_path = sys.argv[1]
    instance_data = tifffile.imread(file_path)

    xy_unit = 0.891
    z_unit = 0.61

    # Use voxel coordinates for mesh/voxelization alignment
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
    shell_volume = mesh_to_shell_volume(
        mesh,
        output_shape=instance_data.shape,
        pitch=1.0,
        shell_only=True,
        thickness=1,
    )

    out_dir = Path(file_path).parent
    filled_tif_path = str(out_dir / "AL_convex_filled.tif")
    shell_tif_path = str(out_dir / "AL_convex_shell.tif")
    tifffile.imwrite(filled_tif_path, mask_to_uint8(filled_volume))
    tifffile.imwrite(shell_tif_path, mask_to_uint8(shell_volume))
    voxel_volume = xy_unit * xy_unit * z_unit
    filled_volume_real = float(filled_volume.sum()) * voxel_volume
    print("Filled volume (real units^3):", filled_volume_real)
    print("Filled volume saved to:", filled_tif_path)
    print("Shell volume saved to:", shell_tif_path)
