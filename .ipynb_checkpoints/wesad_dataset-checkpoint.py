import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import find_peaks

class WESADDataset(Dataset):
    """
    Loads WESAD PPG chunks and calculates HRV metrics on the fly.
    Formats labels as a dictionary for the 4-head MultiTaskPPGNet.
    """
    def __init__(self, chunk_dir):
        # Find all signal files
        self.x_files = sorted(glob.glob(os.path.join(chunk_dir, '*_X.npy')))
        if len(self.x_files) == 0:
            raise FileNotFoundError(f"No _X.npy files found in {chunk_dir}")
        print(f"Loaded {len(self.x_files)} WESAD chunks from {chunk_dir}")

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        x_path = self.x_files[idx]
        
        # 1. Load PPG Signal
        signal = np.load(x_path).astype(np.float32)
        x_tensor = torch.tensor(signal).unsqueeze(0) # Shape: [1, 640]
        
        # 2. Load existing HR and Stress labels
        hr_path = x_path.replace('_X.npy', '_hr_y.npy')
        stress_path = x_path.replace('_X.npy', '_stress_y.npy')
        
        hr_val = float(np.load(hr_path)[0])
        stress_val = int(np.load(stress_path)[0])
        
        # 3. Calculate HRV (RMSSD and SDNN) from the PPG signal
        peaks, _ = find_peaks(signal, distance=15) # distance=15 prevents noise peaks
        
        rmssd = 0.0
        sdnn = 0.0
        
        if len(peaks) > 2:
            rr_intervals = np.diff(peaks) # in samples
            # Convert to seconds (WESAD BVP is 64Hz)
            rr_seconds = rr_intervals / 64.0 
            
            successive_diffs = np.diff(rr_seconds)
            rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
            sdnn = float(np.std(rr_seconds))
            
        # 4. Format labels as dictionary
        # Note: WESAD has no Respiration Rate (RR), so we set it to -1.0
        labels = {
            'hr': torch.tensor(hr_val, dtype=torch.float32),
            'hrv': torch.tensor([rmssd, sdnn], dtype=torch.float32),
            'stress': torch.tensor(stress_val, dtype=torch.long),
            'rr': torch.tensor(-1.0, dtype=torch.float32) 
        }
        
        return x_tensor, labels