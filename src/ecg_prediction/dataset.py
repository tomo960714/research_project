from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset

Split = Literal["train", "validation", "test"]

REQUIRED_COLUMNS = [
    "ecg_id",
    "patient_id",
    "sex",
    "strat_fold",
    "filename_lr"
]

def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """
    Load metadata from the CSV file in the specified data directory.

    Args:
        data_dir (str | Path): The directory containing the metadata CSV file.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / "ptbxl_database.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Exclude records without a valid label or patient identifier.
    metadata = metadata.dropna(subset=["sex", "patient_id"])

    # PTB-XL encoding: 0 = male, 1 = female.
    metadata = metadata[metadata["sex"].isin([0, 1])].copy()
    metadata["sex"] = metadata["sex"].astype("int64")

     # Ensure the fold assignments are valid.
    if not metadata["strat_fold"].isin(range(1, 11)).all():
        raise ValueError("Invalid PTB-XL fold assignments found.")

    if metadata["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ECG identifiers found.")

    return metadata.reset_index(drop=True)

def split_metadata(
    metadata: pd.DataFrame,
    split: Split,
) -> pd.DataFrame:
    """Select records using the official PTB-XL fold assignments."""
    fold_map = {
        "train": list(range(1, 9)),
        "validation": [9],
        "test": [10],
    }

    if split not in fold_map:
        raise ValueError(f"Unknown split: {split}")

    selected = metadata[
        metadata["strat_fold"].isin(fold_map[split])
    ]

    return selected.reset_index(drop=True)

class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for the PTB-XL dataset.

    Args:
        data_dir (str | Path): Directory containing the PTB-XL data.
        split (Split): Which split to use ("train", "validation", or "test").
    """

    def __init__(self, data_dir: str | Path, metadata: pd.DataFrame) -> None:
        self.data_dir = Path(data_dir)
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self,index:int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]

        # WFDB expects the record path without .hea or .dat.
        record_path = self.data_dir / row["filename_lr"]
        signal, _ = wfdb.rdsamp(str(record_path))

        # WFDB returns (time_steps, leads).
        # PyTorch Conv1d expects (leads, time_steps).
        signal = np.asarray(signal, dtype=np.float32).T

        if signal.shape != (12, 1000):
            raise ValueError(
                f"Unexpected ECG shape for {row['ecg_id']}: "
                f"{signal.shape}"
            )

        ecg = torch.from_numpy(signal.copy())
        label = torch.tensor(float(row["sex"]), dtype=torch.float32)

        return ecg, label
    