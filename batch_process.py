import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import mne
from mne.preprocessing.nirs import (
    optical_density,
    beer_lambert_law,
)

# ============================================================
# Paths
# ============================================================

# Root directory containing all subject folders
RAW_DATA_DIR = Path(
    r"Dataset\Raw_Subject_Data\Raw_Subject_Data"
)

# Output directory
OUTPUT_DIR = Path(
    r"outputs"
)

# Figures and reports
FIGURE_DIR = OUTPUT_DIR / "figures"
# REPORT_DIR = OUTPUT_DIR / "reports"

# Create directories
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
# REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Checking Dataset")
print("=" * 70)

print(f"Dataset directory : {RAW_DATA_DIR}")
print(f"Exists            : {RAW_DATA_DIR.exists()}")

if not RAW_DATA_DIR.exists():
    raise FileNotFoundError(
        f"\nDataset directory not found:\n{RAW_DATA_DIR}"
    )

# ============================================================
# Find Subject Folders
# ============================================================

subject_folders = sorted(
    [
        folder
        for folder in RAW_DATA_DIR.iterdir()
        if folder.is_dir() and folder.name.startswith("Subject-")
    ],
    key=lambda folder: int(folder.name.split("-")[1])
)

print("\n" + "=" * 70)
print("Subjects Found")
print("=" * 70)

print(f"Total subject folders found : {len(subject_folders)}\n")

for folder in subject_folders:
    print(f"Subject-{int(folder.name.split('-')[1]):02d}")

# ------------------------------------------------------------
# Report Missing Subject Numbers
# ------------------------------------------------------------

expected_subjects = set(range(1, 20))

available_subjects = {
    int(folder.name.split("-")[1])
    for folder in subject_folders
}

missing_subjects = sorted(expected_subjects - available_subjects)

if missing_subjects:
    print(f"\nMissing subjects : {missing_subjects}")

# ============================================================
# Find SNIRF File for Each Subject
# ============================================================

subjects = []

print("\n" + "=" * 70)
print("Searching for SNIRF Files")
print("=" * 70)

for folder in subject_folders:

    snirf_files = sorted(folder.glob("*.snirf"))

    if len(snirf_files) == 0:
        print(f"[WARNING] No SNIRF file found in {folder.name}")
        continue

    if len(snirf_files) > 1:
        print(f"[WARNING] Multiple SNIRF files found in {folder.name}")
        print(f"Using: {snirf_files[0].name}")

    subjects.append(
        {
            "id": int(folder.name.split("-")[1]),
            "name": folder.name,
            "folder": folder,
            "snirf": snirf_files[0],
        }
    )

print("\n" + "=" * 70)
print("Dataset Summary")
print("=" * 70)

print(f"Valid subjects : {len(subjects)}\n")

for subject in subjects:
    print(
        f"Subject-{subject['id']:02d}  -->  "
        f"{subject['snirf'].name}"
    )
# ============================================================
# Load SNIRF File
# ============================================================

def load_subject(subject):
    """
    Load a single subject's SNIRF file.

    Parameters
    ----------
    subject : dict
        Dictionary containing subject information.

    Returns
    -------
    raw : mne.io.Raw
        Loaded raw SNIRF recording.
    """

    print("\n" + "=" * 70)
    print(f"Loading {subject['name']}")
    print("=" * 70)

    print(f"File : {subject['snirf']}")

    raw = mne.io.read_raw_snirf(
        subject["snirf"],
        preload=True,
        verbose=False,
    )

    print("\nRecording Summary")
    print(raw)

    print(f"Sampling Frequency : {raw.info['sfreq']:.4f} Hz")
    print(f"Channels           : {len(raw.ch_names)}")
    print(f"Duration           : {raw.times[-1]:.2f} s")

    return raw

# ============================================================
# Process One Subject
# ============================================================

def process_subject(subject):
    """
    Process one subject from raw SNIRF data to analysis-ready results.

    Returns
    -------
    dict
        Contains processed data, evoked responses,
        before/after activation maps and plotting information.
    """

    print("\n" + "=" * 70)
    print(f"Processing {subject['name']}")
    print("=" * 70)

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    subject_output = FIGURE_DIR / subject["name"]
    subject_output.mkdir(parents=True, exist_ok=True)

    print(f"Output folder : {subject_output}")

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    raw = load_subject(subject)

    # --------------------------------------------------------
    # Optical Density
    # --------------------------------------------------------

    print("\nConverting to Optical Density...")

    raw_od = optical_density(raw)

    # --------------------------------------------------------
    # Beer-Lambert Law
    # --------------------------------------------------------

    print("Converting to HbO/HbR...")

    raw_hb = beer_lambert_law(raw_od)

    # --------------------------------------------------------
    # Motion correction
    # --------------------------------------------------------

    print("Applying TDDR motion correction...")

    raw_hb = mne.preprocessing.nirs.temporal_derivative_distribution_repair(
        raw_hb
    )

    # --------------------------------------------------------
    # Band-pass filter
    # --------------------------------------------------------

    print("Band-pass filtering (0.01–0.10 Hz)...")

    raw_hb.filter(
        l_freq=0.01,
        h_freq=0.10,
        verbose=False
    )

    print("✓ Preprocessing completed.")

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    print("\nExtracting events...")

    events, event_id = mne.events_from_annotations(raw_hb)

    print("Conditions found:")
    print(event_id)

    print(f"Total events : {len(events)}")

    # --------------------------------------------------------
    # Remove duplicated events
    # --------------------------------------------------------

    events_df = pd.DataFrame(
        events,
        columns=[
            "Sample",
            "Previous",
            "EventID"
        ]
    )

    before = len(events_df)

    events_df = events_df.drop_duplicates(
        subset=["Sample", "EventID"]
    )

    after = len(events_df)

    events = events_df.to_numpy(dtype=int)

    print("\nDuplicate Removal")
    print("-" * 40)
    print(f"Original events : {before}")
    print(f"Remaining       : {after}")
    print(f"Removed         : {before-after}")

    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    print("\nCreating epochs...")

    epochs = mne.Epochs(
        raw_hb,
        events,
        event_id=event_id,
        tmin=-2,
        tmax=15,
        baseline=None,
        preload=True,
        verbose=False
    )

    print("✓ Epochs created")
    print(epochs)

    print("\nTrials per condition")

    for cond in sorted(event_id.keys()):
        print(f"{cond}: {len(epochs[cond])}")

    # --------------------------------------------------------
    # Average epochs
    # --------------------------------------------------------

    print("\nComputing evoked responses...")

    evoked = {
        cond: epochs[cond].average()
        for cond in sorted(event_id.keys())
    }

    print("✓ Evoked responses computed.")

    # --------------------------------------------------------
    # Separate HbO / HbR
    # --------------------------------------------------------

    evoked_hbo = {
        cond: ev.copy().pick("hbo")
        for cond, ev in evoked.items()
    }

    evoked_hbr = {
        cond: ev.copy().pick("hbr")
        for cond, ev in evoked.items()
    }

    print("✓ HbO/HbR separated.")

    print(f"\nHbO channels : {len(evoked_hbo['1'].ch_names)}")
    print(f"HbR channels : {len(evoked_hbr['1'].ch_names)}")

    # --------------------------------------------------------
    # Compute Before / After / Difference
    # --------------------------------------------------------

    print("\nComputing activation maps...")

    before_hbo = {}
    after_hbo = {}
    difference_hbo = {}

    before_hbr = {}
    after_hbr = {}
    difference_hbr = {}

    for cond in sorted(evoked_hbo.keys()):

        times = evoked_hbo[cond].times

        before_idx = np.where(
            (times >= -2) &
            (times <= 0)
        )[0]

        after_idx = np.where(
            (times >= 5) &
            (times <= 10)
        )[0]

        before_hbo[cond] = np.nan_to_num(
            evoked_hbo[cond].data[:, before_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        after_hbo[cond] = np.nan_to_num(
            evoked_hbo[cond].data[:, after_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        difference_hbo[cond] = (
            after_hbo[cond] -
            before_hbo[cond]
        )

        before_hbr[cond] = np.nan_to_num(
            evoked_hbr[cond].data[:, before_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        after_hbr[cond] = np.nan_to_num(
            evoked_hbr[cond].data[:, after_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        difference_hbr[cond] = (
            after_hbr[cond] -
            before_hbr[cond]
        )

    print("✓ Before / After / Difference computed.")

    # --------------------------------------------------------
    # Common colour limits
    # --------------------------------------------------------

    hbo_limit = max(
        np.nanmax(np.abs(v))
        for v in (
            list(before_hbo.values())
            + list(after_hbo.values())
            + list(difference_hbo.values())
        )
    )

    hbr_limit = max(
        np.nanmax(np.abs(v))
        for v in (
            list(before_hbr.values())
            + list(after_hbr.values())
            + list(difference_hbr.values())
        )
    )

    if hbo_limit == 0:
        hbo_limit = 1e-12

    if hbr_limit == 0:
        hbr_limit = 1e-12

    print(f"HbO colour limit : {hbo_limit:.3e}")
    print(f"HbR colour limit : {hbr_limit:.3e}")

    # --------------------------------------------------------
    # Return everything needed later
    # --------------------------------------------------------

    return {

        "subject": subject,
        "subject_output": subject_output,

        "raw": raw,
        "raw_hb": raw_hb,

        "events": events,
        "event_id": event_id,

        "epochs": epochs,

        "evoked": evoked,
        "evoked_hbo": evoked_hbo,
        "evoked_hbr": evoked_hbr,

        "before_hbo": before_hbo,
        "after_hbo": after_hbo,
        "difference_hbo": difference_hbo,

        "before_hbr": before_hbr,
        "after_hbr": after_hbr,
        "difference_hbr": difference_hbr,

        "times": evoked_hbo["1"].times,

        "conditions": sorted(event_id.keys()),

        "hbo_info": evoked_hbo["1"].info,
        "hbr_info": evoked_hbr["1"].info,

        "hbo_limit": hbo_limit,
        "hbr_limit": hbr_limit,
    }
    
# ============================================================
# Save Results
# ============================================================

def save_subject_results(results):
    """
    Save all topography figures for one subject.
    """

    print("\n" + "=" * 70)
    print(f"Saving results for {results['subject']['name']}")
    print("=" * 70)

    subject_output = results["subject_output"]

    conditions = results["conditions"]

    hbo_info = results["hbo_info"]
    hbr_info = results["hbr_info"]

    hbo_limit = results["hbo_limit"]
    hbr_limit = results["hbr_limit"]

    times = results["times"]

    # --------------------------------------------------------
    # Time points for evolution maps
    # --------------------------------------------------------

    time_points = [0, 2, 4, 6, 8, 10, 12, 14]

    # --------------------------------------------------------
    # Process every condition
    # --------------------------------------------------------

    for cond in conditions:

        print(f"\nCondition {cond}")

        cond_folder = subject_output / f"Condition_{cond}"
        cond_folder.mkdir(parents=True, exist_ok=True)

        # ====================================================
        # HbO
        # ====================================================

        try:

            # -------------------------------
            # Before
            # -------------------------------

            before = np.nan_to_num(
                results["before_hbo"][cond]
            )

            evoked = mne.EvokedArray(
                before.reshape(-1, 1),
                hbo_info,
                tmin=0
            )

            fig = evoked.plot_topomap(
                times=0,
                colorbar=True,
                vlim=(-hbo_limit, hbo_limit),
                time_format="",
                show=False
            )

            fig.suptitle(f"Condition {cond} - HbO Before")

            fig.savefig(
                cond_folder / "HbO_Before.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            # -------------------------------
            # After
            # -------------------------------

            after = np.nan_to_num(
                results["after_hbo"][cond]
            )

            evoked = mne.EvokedArray(
                after.reshape(-1, 1),
                hbo_info,
                tmin=0
            )

            fig = evoked.plot_topomap(
                times=0,
                colorbar=True,
                vlim=(-hbo_limit, hbo_limit),
                time_format="",
                show=False
            )

            fig.suptitle(f"Condition {cond} - HbO After")

            fig.savefig(
                cond_folder / "HbO_After.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            # -------------------------------
            # Difference
            # -------------------------------

            diff = np.nan_to_num(
                results["difference_hbo"][cond]
            )

            evoked = mne.EvokedArray(
                diff.reshape(-1, 1),
                hbo_info,
                tmin=0
            )

            fig = evoked.plot_topomap(
                times=0,
                colorbar=True,
                vlim=(-hbo_limit, hbo_limit),
                time_format="",
                show=False
            )

            fig.suptitle(f"Condition {cond} - HbO Difference")

            fig.savefig(
                cond_folder / "HbO_Difference.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            # -------------------------------
            # Time Course
            # -------------------------------

            fig = results["evoked_hbo"][cond].plot_topomap(
                times=time_points,
                colorbar=True,
                vlim=(-hbo_limit, hbo_limit),
                show=False
            )

            fig.suptitle(f"Condition {cond} - HbO Time Course")

            fig.savefig(
                cond_folder / "HbO_TimeCourse.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

        except Exception as e:

            print(f"HbO plotting failed ({cond})")
            print(e)

        # ====================================================
        # HbR
        # ====================================================

        try:

            before = np.nan_to_num(
                results["before_hbr"][cond]
            )

            evoked = mne.EvokedArray(
                before.reshape(-1, 1),
                hbr_info,
                tmin=0
            )

            fig = evoked.plot_topomap(
                times=0,
                colorbar=True,
                vlim=(-hbr_limit, hbr_limit),
                time_format="",
                show=False
            )

            fig.suptitle(f"Condition {cond} - HbR Before")

            fig.savefig(
                cond_folder / "HbR_Before.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            after = np.nan_to_num(
                results["after_hbr"][cond]
            )

            evoked = mne.EvokedArray(
                after.reshape(-1, 1),
                hbr_info,
                tmin=0
            )

            fig = evoked.plot_topomap(
                times=0,
                colorbar=True,
                vlim=(-hbr_limit, hbr_limit),
                time_format="",
                show=False
            )

            fig.suptitle(f"Condition {cond} - HbR After")

            fig.savefig(
                cond_folder / "HbR_After.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            diff = np.nan_to_num(
                results["difference_hbr"][cond]
            )

            evoked = mne.EvokedArray(
                diff.reshape(-1, 1),
                hbr_info,
                tmin=0
            )

            fig = evoked.plot_topomap(
                times=0,
                colorbar=True,
                vlim=(-hbr_limit, hbr_limit),
                time_format="",
                show=False
            )

            fig.suptitle(f"Condition {cond} - HbR Difference")

            fig.savefig(
                cond_folder / "HbR_Difference.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            fig = results["evoked_hbr"][cond].plot_topomap(
                times=time_points,
                colorbar=True,
                vlim=(-hbr_limit, hbr_limit),
                show=False
            )

            fig.suptitle(f"Condition {cond} - HbR Time Course")

            fig.savefig(
                cond_folder / "HbR_TimeCourse.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

        except Exception as e:

            print(f"HbR plotting failed ({cond})")
            print(e)

    print("\n✓ Finished saving subject.")
    
# ============================================================
# Process All Subjects
# ============================================================

for subject in subjects:

    try:

        results = process_subject(subject)

        save_subject_results(results)

    except Exception as e:

        print("\n" + "=" * 70)
        print(f"FAILED : {subject['name']}")
        print("=" * 70)

        print(e)

        continue