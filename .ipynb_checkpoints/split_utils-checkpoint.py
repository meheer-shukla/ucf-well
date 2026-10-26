import os
import re
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Subset


def extract_subject_id(filepath, pattern=r'^([A-Za-z]*\d+)'):
    """
    Pulls the subject identifier from a chunk filename, e.g.
      'subject14_chunk03_X.npy' -> 'subject14'
      'S7_win02_X.npy'          -> 'S7'

    IMPORTANT: adjust `pattern` if your actual filenames don't start with
    the subject id. Print a few of your x_files to confirm the match before
    trusting this on a real run.
    """
    fname = os.path.basename(filepath)
    match = re.match(pattern, fname)
    if not match:
        raise ValueError(
            f"Could not extract subject id from filename: {fname}. "
            f"Update the `pattern` regex in extract_subject_id() to match your naming scheme."
        )
    return match.group(1)


def subject_grouped_split(dataset, test_size=0.2, random_state=42, pattern=r'^([A-Za-z]*\d+)'):
    """
    Splits a dataset into train/val WITHOUT letting any single subject's
    chunks appear in both splits. Replaces torch's random_split, which
    silently leaks subjects across train/val when chunks come from
    continuous per-subject recordings.

    Assumes dataset.x_files is a list of the chunk file paths, in the
    same order the dataset indexes them.
    """
    if not hasattr(dataset, "x_files"):
        raise AttributeError(
            "subject_grouped_split expects dataset.x_files (list of chunk paths). "
            "Add/expose this attribute on your Dataset class."
        )

    groups = [extract_subject_id(f, pattern) for f in dataset.x_files]
    unique_subjects = sorted(set(groups))
    print(f"Found {len(unique_subjects)} unique subjects across {len(groups)} chunks.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))

    train_subjects = {groups[i] for i in train_idx}
    val_subjects = {groups[i] for i in val_idx}
    overlap = train_subjects & val_subjects
    assert len(overlap) == 0, f"Subject leakage detected! Overlapping subjects: {overlap}"

    print(f"Train subjects: {len(train_subjects)} ({len(train_idx)} chunks)")
    print(f"Val subjects:   {len(val_subjects)} ({len(val_idx)} chunks)")

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def save_split_subjects(dataset, train_subset, val_subset, path, pattern=r'^([A-Za-z]*\d+)'):
    """
    Persists which subjects ended up in train vs val, so a later evaluation
    script can guarantee it only scores models on subjects that were never
    used for training/fine-tuning -- this is what closes the leakage bug
    where eval_stage2.py scored on the FULL dataset, ~80% of which Stage 2
    had already been fine-tuned on.
    """
    train_subjects = sorted({extract_subject_id(dataset.x_files[i], pattern) for i in train_subset.indices})
    val_subjects = sorted({extract_subject_id(dataset.x_files[i], pattern) for i in val_subset.indices})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'train_subjects': train_subjects, 'val_subjects': val_subjects}, f, indent=2)
    print(f"Saved split subject lists to {path}")


def load_val_subject_filter(dataset, path, pattern=r'^([A-Za-z]*\d+)'):
    """
    Rebuilds a Subset containing ONLY the chunks belonging to subjects
    marked as val/held-out in the saved split file.
    """
    with open(path) as f:
        split = json.load(f)
    val_subjects = set(split['val_subjects'])
    indices = [i for i, fpath in enumerate(dataset.x_files)
               if extract_subject_id(fpath, pattern) in val_subjects]
    if not indices:
        raise ValueError("No chunks matched the saved held-out subjects -- "
                          "check that `dataset` was built from the same chunk_dir.")
    return Subset(dataset, indices)