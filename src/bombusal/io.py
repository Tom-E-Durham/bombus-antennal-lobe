import os
import csv
import tifffile
import numpy as np
from pathlib import Path

def load_volume(file_path, standardise=True):
    """
    Load a TIFF volume from the specified file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    
    volume = tifffile.imread(file_path)
    print(f'Loaded file: {file_path}')
    print(f'Shape: {volume.shape}')
    print(f'Data type: {volume.dtype}')

    volume = np.asarray(volume, dtype=np.float32)
    
    if not standardise:
        return volume
    
    global_mean = 5685.8
    global_std = 8127.3
    standardised = (volume - global_mean) / global_std
    standardised = ((standardised - standardised.min()) / 
                        (standardised.max() - standardised.min()))
        
    return standardised

def load_dataset(folder_path):
    """
    Load all TIFF volumes from the specified folder path.
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Directory '{folder_path}' does not exist.")
    files = []
    folder_path = Path(folder_path)
    for tif_file in folder_path.rglob("AL*.tif"):
        if tif_file.name in ["AL1.tif", "AL2.tif"]:
            files.append(str(tif_file))
    print(f"[DEBUG INFO]: Loaded {len(files)} files")
    return files

class ConfocalDataset:
    def __init__(self, folder_path, standardise=True):
        self.root = Path(folder_path)
        self.standardise = standardise
        self.info_map = self._load_information()
        self.index = self._build_index()
        self.ids = sorted(self.index.keys())

    def _load_information(self):
        info_path = self.root / "information.csv"
        if not info_path.exists():
            return {}

        def _to_float(value):
            if value is None or value == "" or value == "None":
                return None
            return float(value)

        info_map = {}
        with info_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dataset_id = row.get("sample")
                al_type = row.get("al_type")
                if not dataset_id or not al_type:
                    continue
                info_map.setdefault(dataset_id, {})
                info_map[dataset_id][al_type] = {
                    "gender": row.get("gender"),
                    "total_glomeruli": _to_float(row.get("total_glomeruli")),
                    "z_unit": _to_float(row.get("z_unit")),
                    "xy_unit": _to_float(row.get("xy_unit")),
                }
        return info_map

    def _build_index(self):
        if not self.root.is_dir():
            raise NotADirectoryError(f"Directory '{self.root}' does not exist.")

        index = {}
        for species_dir in self.root.iterdir():
            if not species_dir.is_dir():
                continue
            species = species_dir.name
            for id_dir in species_dir.iterdir():
                if not id_dir.is_dir():
                    continue
                dataset_id = id_dir.name

                al1_path = id_dir / "AL1.tif"
                al2_path = id_dir / "AL2.tif"
                wb_path = id_dir / "wholebrain.tif"

                cache_al1 = (id_dir / "cache_AL1").is_dir()
                cache_al2 = (id_dir / "cache_AL2").is_dir()
                cache_wb = (id_dir / "cache_wb").is_dir()

                index[dataset_id] = {
                    "species": species,
                    "id": dataset_id,
                    "AL1": str(al1_path) if al1_path.exists() else None,
                    "AL2": str(al2_path) if al2_path.exists() else None,
                    "Wholebrain": str(wb_path) if wb_path.exists() else None,
                    "cache": {
                        "AL1": cache_al1,
                        "AL2": cache_al2,
                        "Wholebrain": cache_wb,
                    },
                    "info": self.info_map.get(dataset_id, {}),
                }
        return index

    def get_ids_by_species(self, species):
        return sorted([k for k, v in self.index.items() if v.get("species") == species])

    def get_record_by_id(self, dataset_id):
        if dataset_id not in self.index:
            raise KeyError(f"Dataset id '{dataset_id}' not found.")
        return self.index[dataset_id]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        record = self.get_record_by_id(idx)
        return record