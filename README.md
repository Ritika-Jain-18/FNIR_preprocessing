# fNIRS Emotion Recognition Data Processing

## Overview

This repository contains a complete preprocessing, visualization, and automated reporting pipeline for an fNIRS emotion recognition dataset using **Python** and **MNE-Python**.

The pipeline processes every available subject, performs standard fNIRS preprocessing, extracts task-related haemodynamic responses, generates cortical activation maps, computes activation statistics, and automatically creates a comprehensive Microsoft Word report summarizing the results.

---

## Features

The pipeline performs the following steps:

- Load SNIRF recordings
- Optical Density conversion
- Beer–Lambert Law conversion
- Temporal Derivative Distribution Repair (TDDR) motion correction
- Band-pass filtering (0.01–0.10 Hz)
- Event extraction
- Duplicate event removal
- Epoch creation
- Evoked response computation
- HbO/HbR signal separation
- Automatic generation of:
  - Before (-2 to 0 s) activation maps
  - During (0 to 5 s) activation maps
  - After (5 to 10 s) activation maps
- Trial summary generation
- Mean activation summary generation
- Automatic Word report generation for all processed subjects

---

## Repository Structure

```
FNIR_preprocessing/

├── processing.py
│      Core preprocessing and visualization pipeline
│
├── batch_process.py
│      Batch processing script for all available subjects
│
├── generate_report.py
│      Automatically generates the final Word report
│
├── fNIRS_Assignment_Report.docx
│      Example generated report
│
└── README.md
```

---

## Dataset

The dataset used for this project is publicly available.

**Dataset Link**

https://www.kaggle.com/datasets/aysenureser/fnirs-data-and-analysis-scripts

The dataset is **not included** in this repository.

After downloading the dataset, update the dataset directory inside **processing.py** or **batch_process.py** if required.

---

## Processing Pipeline

```
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
Before / During / After Activation Maps
        │
        ▼
Activation Statistics
        │
        ▼
Automatic Word Report Generation
```

---

## Output Structure

```
outputs/

├── data_figures/
│
│   ├── Subject-1/
│   │
│   ├── Condition_1/
│   │      ├── HbO_Before.png
│   │      ├── HbO_During.png
│   │      ├── HbO_After.png
│   │      ├── HbR_Before.png
│   │      ├── HbR_During.png
│   │      └── HbR_After.png
│   │
│   ├── Condition_2/
│   ├── Condition_4/
│   ├── Condition_6/
│   │
│   ├── trial_summary.csv
│   └── activation_summary.csv
│
├── Subject-2/
├── ...
│
└── fNIRS_Assignment_Report.docx
```

---

## Generated Report

The automatically generated report includes:

- Subject-wise processing results
- HbO activation maps
- HbR activation maps
- Before, During and After activation maps
- All annotated experimental conditions (1, 2, 4 and 6)
- Trial Summary tables
- Activation Summary tables
- Overall processing summary

---

## Requirements

Python **3.10+**

Main libraries:

```
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

## Running the Pipeline

### Process all subjects

```bash
python batch_process.py
```

### Generate the final report

```bash
python generate_report.py
```

The report will be generated as:

```
outputs/fNIRS_Assignment_Report.docx
```

---

## Example Outputs

An example report is included in this repository:

- **fNIRS_Assignment_Report.docx**

Complete generated outputs are available here:

**Google Drive**

https://drive.google.com/file/d/1wKekIGkBKjFKwzZhlwfN8stF6SKi7ViS/view?usp=drive_link

---

## Notes

- Subjects **6** and **9** were not present in the provided dataset.
- The pipeline automatically processes every available subject.
- Duplicate event annotations are removed before epoch creation.
- Trial counts and average haemodynamic activation values are automatically computed.
- Some recordings exhibit minimal haemodynamic activity due to signal quality, resulting in nearly uniform activation maps.

---

## Author

**Ritika Jain**

BS-MS (Data Science and Engineering)

Indian Institute of Science Education and Research (IISER) Bhopal
