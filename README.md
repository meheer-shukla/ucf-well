# UCF-Well: Contactless Wellness Monitoring Pipeline

This repository contains the code and cached features for the paper:

**"A Contactless Wellness Pipeline with a Controlled Study of Cross-Modal HRV Adaptation for Stress Detection"**
*Meheer Shukla, Dr. Gopal Gawande*

## Overview
This project extracts physiological signals (HR, HRV) from facial video using remote photoplethysmography (rPPG). It includes stress classification trained on WESAD and evaluated on a calibrated synthetic-rPPG target, along with an emotion classifier trained on AffectNet.

## Reproducing the results
1. Install requirements: `pip install -r backend/requirements.txt`
2. Extract features: run scripts in `scripts/` sequentially (`step3` → `step7` → `step5`).
3. Compile the paper: navigate to `paper/` and compile `main.tex` using `pdflatex`.

*Note: The raw datasets (WESAD, UBFC, AffectNet) are not included due to size. Please download them from their original sources.*