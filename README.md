# fNIRS Emotion Recognition Data Processing

## Overview

This repository contains a complete preprocessing, visualization, and automated reporting pipeline for an fNIRS emotion recognition dataset using **Python** and **MNE-Python**.

The project includes two preprocessing pipelines:

- **Original Pipeline (`batch_process.py`)**
  - Generates **Before**, **After**, and **Difference** activation maps together with HbO/HbR time-course visualizations.

- **Enhanced Pipeline (`processing.py`)**
  - Generates **Before**, **During**, and **After** activation maps together with HbO/HbR time-course visualizations, computes activation statistics, and automatically generates a comprehensive Microsoft Word report.

---

# Features

## Original Processing Pipeline (`batch_process.py`)

- Load SNIRF recordings
- Optical Density conversion
- Beer–Lambert Law conversion
- Temporal Derivative Distribution Repair (TDDR) motion correction
- Band-pass filtering (0.01–0.10 Hz)
- Event extraction
- Duplicate event removal
- Epoch creation
- Evoked response computation
- HbO/HbR separation
- Before activation maps
- After activation maps
- Difference (After − Before) activation maps
- HbO/HbR time-course topographic maps

---

## Enhanced Processing Pipeline (`processing.py`)

Includes all preprocessing steps above and additionally generates:

- Before (-2 to 0 s) activation maps
- During (0 to 5 s) activation maps
- After (5 to 10 s) activation maps
- HbO/HbR time-course topographic maps
- Trial Summary tables
- Activation Summary tables
- Outputs compatible with automatic report generation

---

## Report Generation (`generate_report.py`)

Automatically generates a structured Microsoft Word report containing:

- Subject-wise results
- HbO activation maps
- HbR activation maps
- Before, During and After activation maps
- Trial Summary tables
- Activation Summary tables
- Overall processing summary

---

# Repository Structure

```text
FNIR_preprocessing/

├── processing.py
│      Enhanced preprocessing pipeline
│      • Before / During / After activation maps
│      • HbO/HbR time-course maps
│      • Trial Summary
│      • Activation Summary
│
├── batch_process.py
│      Original preprocessing pipeline
│      • Before / After / Difference maps
│      • HbO/HbR time-course maps
│
├── generate_report.py
│      Automatically generates the final Word report
│      from the outputs produced by processing.py
│
├── fNIRS_Assignment_Report.docx
│      Example generated report
│
└── README.md
```

---

# Dataset

The dataset used for this project is publicly available.

**Dataset**

https://www.kaggle.com/datasets/aysenureser/fnirs-data-and-analysis-scripts

The dataset is **not included** in this repository.

After downloading the dataset, update the dataset path inside the processing script before execution.

---

# Processing Pipeline

```text
Raw SNIRF Recording
        │
        ▼
Optical Density
        │
        ▼
Beer–Lambert Law
        │
        ▼
TDDR Motion Correction
        │
        ▼
Band-pass Filtering (0.01–0.10 Hz)
        │
        ▼
Event Extraction
        │
        ▼
Duplicate Event Removal
        │
        ▼
Epoch Creation (-2 s to 15 s)
        │
        ▼
Evoked Response Computation
        │
        ▼
HbO / HbR Separation
        │
        ▼
Before / During / After
Activation Maps
        │
        ▼
Activation Statistics
        │
        ▼
Automatic Word Report Generation
```

---

# Output Structure

```text
outputs/

├── data_figures/
│
├── Subject-1/
│   ├── Condition_1/
│   │   ├── HbO_Before.png
│   │   ├── HbO_During.png
│   │   ├── HbO_After.png
│   │   ├── HbR_Before.png
│   │   ├── HbR_During.png
│   │   ├── HbR_After.png
│   │   ├── Trial Summary
│   │   └── Activation Summary
│   │
│   ├── Condition_2/
│   ├── Condition_4/
│   └── Condition_6/
│
├── Subject-2/
├── ...
│
└── fNIRS_Assignment_Report.docx
```

---

# Generated Report

The automatically generated report includes:

- Subject-wise processing results
- HbO activation maps
- HbR activation maps
- Before, During and After activation maps
- Four annotated experimental conditions (1, 2, 4 and 6)
- Trial Summary tables
- Activation Summary tables
- Overall report summary

---

# Requirements

Python **3.10+**

Main libraries:

```text
numpy
pandas
matplotlib
mne
python-docx
```

Install dependencies using:

```bash
pip install numpy pandas matplotlib mne python-docx
```

---

# Running the Pipeline

## Original Pipeline

Generate:

- Before activation maps
- After activation maps
- Difference maps
- HbO/HbR time-course maps

```bash
python batch_process.py
```

---

## Enhanced Pipeline

Generate:

- Before activation maps
- During activation maps
- After activation maps
- HbO/HbR time-course maps
- Trial Summary tables
- Activation Summary tables

```bash
python processing.py
```

---

## Generate the Report

After running `processing.py`:

```bash
python generate_report.py
```

The report will be saved as:

```text
outputs/fNIRS_Assignment_Report.docx
```

---

# Example Outputs

The repository includes an example report:

- **fNIRS_Assignment_Report.docx**

Complete generated outputs are available here:

**Google Drive**

https://drive.google.com/file/d/1wKekIGkBKjFKwzZhlwfN8stF6SKi7ViS/view?usp=drive_link

---

# Notes

- Subjects **6** and **9** were not present in the provided dataset.
- The pipeline automatically detects and processes every available subject.
- Duplicate event annotations are removed before epoch creation.
- Trial counts and average haemodynamic responses are automatically summarized.
- Some recordings exhibit minimal haemodynamic activity due to signal quality, resulting in nearly uniform activation maps.

---

# Author

**Ritika Jain**

BS-MS (Data Science and Engineering)

Indian Institute of Science Education and Research (IISER) Bhopal
