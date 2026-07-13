# fNIRS Emotion Recognition Data Processing

## Overview

This repository contains a complete preprocessing and visualization pipeline for an fNIRS emotion recognition dataset using Python and MNE-Python.

The pipeline processes all available subjects, performs standard fNIRS preprocessing, extracts task-related epochs, computes evoked haemodynamic responses, and generates topographic visualizations for HbO and HbR signals.

---

## Features

The processing pipeline includes:

- Loading SNIRF recordings
- Optical Density conversion
- Beer-Lambert Law conversion
- Temporal Derivative Distribution Repair (TDDR) motion correction
- Band-pass filtering (0.01–0.10 Hz)
- Event extraction
- Duplicate event removal
- Epoch creation
- Evoked response computation
- HbO/HbR separation
- Baseline (Before) activation maps
- Task (After) activation maps
- Difference maps (After − Before)
- Time-resolved topographic maps

---

## Dataset

The dataset used for this project is publicly available.

Dataset Link:

(https://www.kaggle.com/datasets/aysenureser/fnirs-data-and-analysis-scripts)

The dataset is **not included** in this repository.

After downloading the dataset, update the following path inside `batch_process.py`:

```python
RAW_DATA_DIR = Path(r"YOUR_DATASET_PATH")
```

---

## Project Structure

```
batch_process.py

outputs/
    Subject-1/
        Condition_1/
        Condition_2/
        Condition_4/
        Condition_6/

    Subject-2/
    ...

```

Each condition contains:

- HbO Before map
- HbO After map
- HbO Difference map
- HbO Time-course map
- HbR Before map
- HbR After map
- HbR Difference map
- HbR Time-course map

---

## Requirements

Python 3.10+

Main libraries:

- numpy
- pandas
- matplotlib
- mne

Install dependencies using:

```bash
pip install numpy pandas matplotlib mne
```

---

## Processing Pipeline

```
SNIRF
   │
   ▼
Optical Density
   │
   ▼
Beer-Lambert Law
   │
   ▼
TDDR Motion Correction
   │
   ▼
Band-pass Filter
   │
   ▼
Event Extraction
   │
   ▼
Epoching
   │
   ▼
Evoked Responses
   │
   ▼
HbO / HbR Separation
   │
   ▼
Before / After / Difference Maps
   │
   ▼
Time-resolved Topographic Maps
```

---

## Output

For every subject and every experimental condition, the pipeline generates:

- Before activation map
- After activation map
- Difference map
- Time-course topographic maps

---

## Notes

- Subjects 6 and 9 were not present in the provided dataset.
- Some recordings show minimal haemodynamic activity due to signal quality, resulting in nearly uniform activation maps.
- The pipeline processes all available subjects automatically.

---

## Author

Ritika Jain

BS-MS (Data Science and Engineering)

Indian Institute of Science Education and Research (IISER) Bhopal
