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
    "Dataset",
    "Raw_Subject_Data",
    "Raw_Subject_Data",
)

# Output directories
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "data_figures"
REPORT_DIR = OUTPUT_DIR / "reports"

# Create output directories
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


print_header("Checking Dataset")

print(f"Dataset directory : {RAW_DATA_DIR}")
print(f"Exists            : {RAW_DATA_DIR.is_dir()}")

if not RAW_DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset directory does not exist:\n{RAW_DATA_DIR}\n"
        "Please download the dataset and update RAW_DATA_DIR."
    )
    
# ============================================================
# Find Subject Folders
# ============================================================

def get_subject_id(folder: Path) -> int:
    """Extract numeric subject ID from a folder name."""
    try:
        return int(folder.name.split("-")[1])
    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid subject folder name: {folder.name}"
        )


subject_folders = sorted(
    (
        folder
        for folder in RAW_DATA_DIR.iterdir()
        if folder.is_dir() and folder.name.startswith("Subject-")
    ),
    key=get_subject_id,
)

print_header("Subjects Found")

if not subject_folders:
    raise RuntimeError(
        "No subject folders were found in the dataset directory."
    )

print(f"Total subject folders found : {len(subject_folders)}\n")

for folder in subject_folders:
    print(f"Subject-{get_subject_id(folder):02d}")

# ------------------------------------------------------------
# Report Missing Subject Numbers
# ------------------------------------------------------------

expected_subjects = set(range(1, 20))

available_subjects = {
    get_subject_id(folder)
    for folder in subject_folders
}

missing_subjects = sorted(expected_subjects - available_subjects)

if missing_subjects:
    print(
        "\nMissing subject IDs : "
        + ", ".join(map(str, missing_subjects))
    )

# ============================================================
# Find SNIRF File for Each Subject
# ============================================================

subjects = []

print_header("Searching for SNIRF Files")

for folder in subject_folders:

    snirf_files = sorted(folder.glob("*.snirf"))

    if not snirf_files:
        print(f"[WARNING] No SNIRF file found in {folder.name}")
        continue

    if len(snirf_files) > 1:
        print(f"[WARNING] Multiple SNIRF files found in {folder.name}")
        print(f"Using first file: {snirf_files[0].name}")

    subject_id = get_subject_id(folder)

    subjects.append(
        {
            "id": subject_id,
            "name": folder.name,
            "folder": folder,
            "snirf": snirf_files[0],
        }
    )

print_header("Dataset Summary")

print(f"Valid subjects : {len(subjects)}\n")

for subject in subjects:
    print(
        f"Subject-{subject['id']:02d} --> "
        f"{subject['snirf'].name}"
    )
    
# ============================================================
# Load SNIRF File
# ============================================================

def load_subject(subject: dict) -> mne.io.BaseRaw:
    """
    Load a subject's SNIRF recording into an MNE Raw object.

    Parameters
    ----------
    subject : dict
        Dictionary containing subject metadata.

    Returns
    -------
    mne.io.BaseRaw
        Loaded raw SNIRF recording.
    """

    print_header(f"Loading {subject['name']}")

    print(f"File : {subject['snirf']}")

    try:
        raw = mne.io.read_raw_snirf(
            subject["snirf"],
            preload=True,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SNIRF file:\n{subject['snirf']}"
        ) from e

    if raw.n_times == 0:
        raise ValueError(
            f"{subject['name']} contains no recording data."
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

def process_subject(subject: dict) -> dict:
    """
    Process one subject from raw SNIRF data to analysis-ready
    haemodynamic responses.

    Performs preprocessing, event extraction, epoching,
    evoked response computation, and computes activation
    maps for subsequent analysis.

    Parameters
    ----------
    subject : dict
        Dictionary containing subject metadata.

    Returns
    -------
    dict
        Dictionary containing processed data, evoked responses,
        activation maps, and plotting information.
    """

    print_header(f"Processing {subject['name']}")

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
    # Preprocessing
    # --------------------------------------------------------

    try:
        print("\nConverting to Optical Density...")
        raw_od = optical_density(raw)

        print("Converting to HbO/HbR...")
        raw_hb = beer_lambert_law(raw_od)

        print("Applying TDDR motion correction...")
        raw_hb = (
            mne.preprocessing.nirs
            .temporal_derivative_distribution_repair(raw_hb)
        )

        print("Band-pass filtering (0.01–0.10 Hz)...")
        raw_hb.filter(
            l_freq=0.01,
            h_freq=0.10,
            verbose=False,
        )

    except Exception as e:
        raise RuntimeError(
            f"Preprocessing failed for {subject['name']}"
        ) from e

    print("✓ Preprocessing completed.")
    
    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    print("\nExtracting events...")

    events, event_id = mne.events_from_annotations(raw_hb)

    print("Conditions found:")
    print(event_id)

    print(f"Total events : {len(events)}")

    expected_conditions = {"1", "2", "4", "6"}
    missing_conditions = expected_conditions - set(event_id.keys())

    if missing_conditions:
        print(
            f"Warning: Missing conditions: "
            f"{sorted(missing_conditions)}"
        )

    # --------------------------------------------------------
    # Remove duplicated events
    # --------------------------------------------------------

    events_df = pd.DataFrame(
        events,
        columns=[
            "Sample",
            "Previous",
            "EventID",
        ],
    )

    original_events = len(events_df)

    duplicates = events_df.duplicated(
        subset=["Sample", "EventID"]
    ).sum()

    if duplicates:
        events_df = events_df.drop_duplicates(
            subset=["Sample", "EventID"]
        )

    remaining_events = len(events_df)

    events = events_df.to_numpy(dtype=int)

    print("\nDuplicate Removal")
    print("-" * 40)
    print(f"Original events : {original_events}")
    print(f"Remaining       : {remaining_events}")
    print(f"Removed         : {original_events - remaining_events}")

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
        reject_by_annotation=True,
        verbose=False,
    )

    print("✓ Epochs created")
    print(epochs)

    print("\nTrials per condition")

    trial_summary = []

    for cond in sorted(event_id.keys()):
        n_trials = len(epochs[cond])
        print(f"{cond}: {n_trials}")

        trial_summary.append(
            {
                "Condition": cond,
                "Trials": n_trials,
            }
        )

    trial_summary = pd.DataFrame(trial_summary)

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
    # Compute Before / During / After Activation Maps
    # --------------------------------------------------------

    print("\nComputing activation maps...")

    before_hbo = {}
    during_hbo = {}
    after_hbo = {}

    before_hbr = {}
    during_hbr = {}
    after_hbr = {}

    summary_rows = []

    for cond in sorted(evoked_hbo.keys()):

        times = evoked_hbo[cond].times

        before_idx = np.where(
            (times >= -2) & (times < 0)
        )[0]

        during_idx = np.where(
            (times >= 0) & (times <= 5)
        )[0]

        after_idx = np.where(
            (times > 5) & (times <= 10)
        )[0]

        if len(before_idx) == 0:
            raise RuntimeError("Before window contains no samples.")

        if len(during_idx) == 0:
            raise RuntimeError("During window contains no samples.")

        if len(after_idx) == 0:
            raise RuntimeError("After window contains no samples.")

        before_hbo[cond] = np.nan_to_num(
            evoked_hbo[cond].data[:, before_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        during_hbo[cond] = np.nan_to_num(
            evoked_hbo[cond].data[:, during_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        after_hbo[cond] = np.nan_to_num(
            evoked_hbo[cond].data[:, after_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        before_hbr[cond] = np.nan_to_num(
            evoked_hbr[cond].data[:, before_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        during_hbr[cond] = np.nan_to_num(
            evoked_hbr[cond].data[:, during_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        after_hbr[cond] = np.nan_to_num(
            evoked_hbr[cond].data[:, after_idx].mean(axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        summary_rows.append(
            {
                "Condition": cond,
                "HbO Before": before_hbo[cond].mean(),
                "HbO During": during_hbo[cond].mean(),
                "HbO After": after_hbo[cond].mean(),
                "HbR Before": before_hbr[cond].mean(),
                "HbR During": during_hbr[cond].mean(),
                "HbR After": after_hbr[cond].mean(),
            }
        )

    activation_summary = pd.DataFrame(summary_rows)

    print("✓ Before / During / After activation maps computed.")
    
    # --------------------------------------------------------
    # Common colour limits
    # --------------------------------------------------------

    hbo_limit = max(
        np.nanmax(np.abs(v))
        for v in (
            list(before_hbo.values())
            + list(during_hbo.values())
            + list(after_hbo.values())
        )
    )

    hbr_limit = max(
        np.nanmax(np.abs(v))
        for v in (
            list(before_hbr.values())
            + list(during_hbr.values())
            + list(after_hbr.values())
        )
    )

    if hbo_limit <= 0:
        hbo_limit = 1e-12

    if hbr_limit <= 0:
        hbr_limit = 1e-12

    print(f"HbO colour limit : {hbo_limit:.3e}")
    print(f"HbR colour limit : {hbr_limit:.3e}")

    # --------------------------------------------------------
    # Return everything needed later
    # --------------------------------------------------------

    first_condition = sorted(evoked_hbo.keys())[0]

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
        "during_hbo": during_hbo,
        "after_hbo": after_hbo,

        "before_hbr": before_hbr,
        "during_hbr": during_hbr,
        "after_hbr": after_hbr,

        "activation_summary": activation_summary,
        "trial_summary": trial_summary,

        "times": evoked_hbo[first_condition].times,

        "conditions": sorted(event_id.keys()),

        "hbo_info": evoked_hbo[first_condition].info,
        "hbr_info": evoked_hbr[first_condition].info,

        "hbo_limit": hbo_limit,
        "hbr_limit": hbr_limit,
    }
    

def save_single_topomap(
    data: np.ndarray,
    info,
    title: str,
    filename: Path,
    limit: float,
) -> None:
    """
    Save a single topographic activation map.

    The data are scaled to micro-molar (µM) for visualization only.
    The underlying analysis remains unchanged.
    """

    # --------------------------------------------------------
    # Scale data to µM for display
    # --------------------------------------------------------

    display_data = np.nan_to_num(data) * 1e6
    display_limit = limit * 1e6

    # --------------------------------------------------------
    # Create Evoked object
    # --------------------------------------------------------

    evoked = mne.EvokedArray(
        display_data.reshape(-1, 1),
        info,
        tmin=0,
    )

    # --------------------------------------------------------
    # Plot topomap
    # --------------------------------------------------------

    fig = evoked.plot_topomap(
        times=0,
        colorbar=True,
        vlim=(-display_limit, display_limit),
        cmap="RdBu_r",          # Standard neuroimaging colormap
        time_format="",
        show=False,
    )

    # --------------------------------------------------------
    # Improve title
    # --------------------------------------------------------

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
    
# ============================================================
# Save Results
# ============================================================

def save_subject_results(results: dict) -> None:
    """
    Save all generated figures and summary files for one subject.
    """

    print_header(
        f"Saving results for {results['subject']['name']}"
    )

    subject_output = results["subject_output"]

    conditions = results["conditions"]

    hbo_info = results["hbo_info"]
    hbr_info = results["hbr_info"]

    hbo_limit = results["hbo_limit"]
    hbr_limit = results["hbr_limit"]

    times = results["times"]

    # --------------------------------------------------------
    # Save summary tables
    # --------------------------------------------------------

    results["activation_summary"].to_csv(
        subject_output / "activation_summary.csv",
        index=False,
    )

    results["trial_summary"].to_csv(
        subject_output / "trial_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Time points for evolution maps
    # --------------------------------------------------------

    time_points = [
        t
        for t in [-2, 0, 2, 4, 6, 8, 10, 12, 14]
        if times.min() <= t <= times.max()
    ]
    
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
    
            save_single_topomap(
                results["before_hbo"][cond],
                hbo_info,
                f"Condition {cond} - HbO Before",
                cond_folder / "HbO_Before.png",
                hbo_limit,
            )

            save_single_topomap(
                results["during_hbo"][cond],
                hbo_info,
                f"Condition {cond} - HbO During",
                cond_folder / "HbO_During.png",
                hbo_limit,
            )
    
            save_single_topomap(
                results["after_hbo"][cond],
                hbo_info,
                f"Condition {cond} - HbO After",
                cond_folder / "HbO_After.png",
                hbo_limit,
            )
    
            # fig = results["evoked_hbo"][cond].plot_topomap(
            #     times=time_points,
            #     colorbar=True,
            #     vlim=(-hbo_limit, hbo_limit),
            #     show=False,
            # )
            
            fig = results["evoked_hbo"][cond].copy()

            fig.apply_function(lambda x: x * 1e6)

            fig = fig.plot_topomap(
                times=time_points,
                colorbar=True,
                vlim=(-hbo_limit * 1e6, hbo_limit * 1e6),
                cmap="RdBu_r",
                show=False,
            )
    
            fig.suptitle(
                f"Condition {cond} - HbO Time-course"
            )
    
            fig.savefig(
                cond_folder / "HbO_TimeCourse.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

        except Exception as e:

            print(f"[ERROR] HbO plotting failed ({cond})")
            print(e)

        # ====================================================
        # HbR
        # ====================================================

        try:

            save_single_topomap(
                results["before_hbr"][cond],
                hbr_info,
                f"Condition {cond} - HbR Before",
                cond_folder / "HbR_Before.png",
                hbr_limit,
            )

            save_single_topomap(
                results["during_hbr"][cond],
                hbr_info,
                f"Condition {cond} - HbR During",
                cond_folder / "HbR_During.png",
                hbr_limit,
            )

            save_single_topomap(
                results["after_hbr"][cond],
                hbr_info,
                f"Condition {cond} - HbR After",
                cond_folder / "HbR_After.png",
                hbr_limit,
            )

            # fig = results["evoked_hbr"][cond].plot_topomap(
            #     times=time_points,
            #     colorbar=True,
            #     vlim=(-hbr_limit, hbr_limit),
            #     show=False,
            # )
            
            fig = results["evoked_hbr"][cond].copy()

            fig.apply_function(lambda x: x * 1e6)

            fig = fig.plot_topomap(
                times=time_points,
                colorbar=True,
                vlim=(-hbr_limit * 1e6, hbr_limit * 1e6),
                cmap="RdBu_r",
                show=False,
            )

            fig.suptitle(
                f"Condition {cond} - HbR Time-course"
            )

            fig.savefig(
                cond_folder / "HbR_TimeCourse.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

        except Exception as e:

            print(f"[ERROR] HbR plotting failed ({cond})")
            print(e)

    print("\n✓ Finished saving subject.")
    
# ============================================================
# Process All Subjects
# ============================================================

import traceback

successful_subjects = []
failed_subjects = []

for subject in subjects:

    try:

        results = process_subject(subject)

        save_subject_results(results)

        successful_subjects.append(subject["id"])

    except Exception:

        failed_subjects.append(subject["id"])

        print_header(f"FAILED : {subject['name']}")

        traceback.print_exc()

        continue

print_header("Processing Summary")

print(f"Successfully processed : {len(successful_subjects)}")
print(f"Failed                : {len(failed_subjects)}")

if successful_subjects:
    print(f"Processed subjects : {successful_subjects}")

if failed_subjects:
    print(f"Failed subjects    : {failed_subjects}")

print("\n✓ Processing completed.")