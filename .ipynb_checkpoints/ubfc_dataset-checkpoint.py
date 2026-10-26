import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from rppg_pos import pos_algorithm


class UBFCDataset(Dataset):
    """
    Loads UBFC video chunks, extracts a POS signal, and formats labels
    for the 4-head MultiTaskPPGNet.

    CHANGES from the original version:
      1. ROI: the previous version did a blind center 32x32 crop of the
         face frame, which likely lands on nose/mouth rather than
         forehead/cheeks -- this does NOT match Methodology Section 4.3
         (MediaPipe landmark-based forehead+cheek ROI). Until the raw
         video + landmarks are re-extracted, this uses a heuristic
         whole-face average (minus a 10% border) as an honest stand-in.
         REPLACE _extract_pulse() with real landmark-based ROI masking
         as soon as you have per-frame landmark coordinates saved.
      2. Chunks where POS extraction fails are DROPPED (not zero-padded),
         and the drop count is logged, so garbage signals don't silently
         pollute training/eval.
      3. Pulses are extracted and cached once in __init__ instead of
         recomputed every __getitem__ call.
    """

    def __init__(self, chunk_dir, target_len=640, fps=30, min_pulse_len=30):
        x_files = sorted(glob.glob(os.path.join(chunk_dir, '*_X.npy')))
        if len(x_files) == 0:
            raise FileNotFoundError(f"No _X.npy files found in {chunk_dir}")

        self.samples = []  # list of (x_path, resampled_pulse)
        n_failed = 0

        for x_path in x_files:
            pulse = self._extract_pulse(x_path, fps)

            if pulse is None or len(pulse) < min_pulse_len:
                n_failed += 1
                continue

            pulse = (pulse - np.mean(pulse)) / (np.std(pulse) + 1e-6)
            pulse_resampled = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(pulse)),
                pulse
            ).astype(np.float32)

            self.samples.append((x_path, pulse_resampled))

        if n_failed:
            print(f"WARNING: dropped {n_failed}/{len(x_files)} chunks "
                  f"where POS extraction failed (kept {len(self.samples)}).")

        print(f"Loaded {len(self.samples)} usable UBFC video chunks from {chunk_dir}")

        # exposed for subject_grouped_split() in split_utils.py
        self.x_files = [s[0] for s in self.samples]

    @staticmethod
    def _extract_pulse(x_path, fps):
        video = np.load(x_path).astype(np.float32)  # [T, H, W, 3]

        # Heuristic whole-face ROI (10% border trimmed to reduce background
        # bleed). See class docstring -- replace with real landmark ROI
        # (forehead + cheeks) when available.
        h, w = video.shape[1], video.shape[2]
        mh, mw = int(h * 0.1), int(w * 0.1)
        face_video = video[:, mh:h - mh, mw:w - mw, :]

        rgb_signal = face_video.mean(axis=(1, 2))  # [T, 3]
        return pos_algorithm(rgb_signal, fps=fps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_path, pulse_resampled = self.samples[idx]
        x_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0)

        y_path = x_path.replace('_X.npy', '_y.npy')
        labels_dict = np.load(y_path, allow_pickle=True).item()

        # NOTE: UBFC-rPPG has no real stress-induction protocol or
        # respiration ground truth. If 'stress'/'rr' in these y-files are
        # placeholders, never report metrics computed against them.
        labels = {
            'hr': torch.tensor(labels_dict['hr'], dtype=torch.float32),
            'hrv': torch.tensor(labels_dict['hrv'], dtype=torch.float32),
            'stress': torch.tensor(labels_dict['stress'], dtype=torch.long),
            'rr': torch.tensor(labels_dict['rr'], dtype=torch.float32)
        }

        return x_tensor, labels