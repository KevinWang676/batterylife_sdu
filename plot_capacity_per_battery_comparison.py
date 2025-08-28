#!/usr/bin/env python3
"""
Create side-by-side capacity trajectory plots per battery:
 - Left: Before preprocessing (raw from CSVs), Discharge_Capacity(Ah) vs Observed Cycle Index
 - Right: After preprocessing (from pickles), Discharge_Capacity(Ah) vs Processed Cycle Index
Exclude Battery_ID 73, 74, 75 from plotting.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

RAW_DIR = Path("/Users/kevinwang/Downloads/14859405/Primary_use_phase")
PROCESSED_DIR = Path("/Users/kevinwang/Downloads/BatteryLife/processed_primary_use_phase_08_24")  # diagnose-only
PROCESSED_COMPLETE_DIR = Path("/Users/kevinwang/Downloads/BatteryLife/processed_primary_use_phase_08_24_complete")
OUT_DIR = Path("/Users/kevinwang/Downloads/BatteryLife/plots/comparison_capacity_per_battery_08_28")

EXCLUDE_BATTERIES = set()

plt.style.use('seaborn-v0_8')


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_trajectories() -> Dict[int, Tuple[List[int], List[float]]]:
    """Return mapping Battery_ID -> (observed_cycles, capacities)."""
    csv_files = sorted(RAW_DIR.glob('*.csv'))
    battery_groups: Dict[int, List[pd.DataFrame]] = {}
    for csv_file in tqdm(csv_files, desc='Reading raw CSVs'):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        if 'Battery_ID' not in df.columns or 'Discharge_Capacity(Ah)' not in df.columns:
            continue
        for bid, bdf in df.groupby('Battery_ID'):
            try:
                bid_int = int(bid)
            except Exception:
                continue
            if bid_int in EXCLUDE_BATTERIES:
                continue
            battery_groups.setdefault(bid_int, []).append(bdf)

    raw_map: Dict[int, Tuple[List[int], List[float]]] = {}
    for bid, parts in battery_groups.items():
        bdf = pd.concat(parts, ignore_index=True)
        # Sort by time to ensure ordering
        bdf = bdf.sort_values(['Test_Time(s)'])
        # Max discharge capacity per cycle
        grouped = (
            bdf.groupby('Cycle_Index', sort=True)['Discharge_Capacity(Ah)']
            .max()
            .dropna()
            .reset_index()
        )
        # Observed cycle index starting at 1 in sorted order
        grouped = grouped.sort_values('Cycle_Index').reset_index(drop=True)
        observed_cycles = list(np.arange(1, len(grouped) + 1))
        capacities = grouped['Discharge_Capacity(Ah)'].astype(float).tolist()
        if capacities:
            raw_map[bid] = (observed_cycles, capacities)
    return raw_map


def load_processed_trajectories_from(base_dir: Path) -> Dict[int, Tuple[str, List[int], List[float], List[int], List[int]]]:
    """Return mapping Battery_ID -> (cell_id, processed_cycles, capacities, hardcoded_removed_indices, median_removed_indices).

    hardcoded_removed_indices and median_removed_indices are 1-based indices in the raw cycle space as persisted by the preprocessor.
    """
    pkl_files = sorted(base_dir.glob('*.pkl'))
    proc_map: Dict[int, Tuple[str, List[int], List[float], List[int], List[int]]] = {}
    for pkl in tqdm(pkl_files, desc=f'Reading processed pickles from {base_dir.name}'):
        try:
            with open(pkl, 'rb') as f:
                obj = pickle.load(f)
        except Exception:
            continue
        stem = pkl.stem  # e.g., SDU_Battery_85
        try:
            bid = int(stem.split('_')[-1])
        except Exception:
            continue
        if bid in EXCLUDE_BATTERIES:
            continue
        hardcoded_removed_indices: List[int] = []
        median_removed_indices: List[int] = []
        if isinstance(obj, dict):
            cell_id = obj.get('cell_id', stem)
            cycle_data = obj['cycle_data']
            hardcoded_removed_indices = list(obj.get('hardcoded_removed_indices', []))
            median_removed_indices = list(obj.get('median_removed_indices', []))
        else:
            cell_id = getattr(obj, 'cell_id', stem)
            cycle_data = obj.cycle_data
            hardcoded_removed_indices = list(getattr(obj, 'hardcoded_removed_indices', []) or [])
            median_removed_indices = list(getattr(obj, 'median_removed_indices', []) or [])
        cycles, caps = [], []
        for cyc in cycle_data:
            if isinstance(cyc, dict):
                cycles.append(int(cyc['cycle_number']))
                dc = cyc['discharge_capacity_in_Ah'] or []
                caps.append(float(max(dc)) if len(dc) > 0 else 0.0)
            else:
                cycles.append(int(cyc.cycle_number))
                dc = getattr(cyc, 'discharge_capacity_in_Ah', []) or []
                caps.append(float(max(dc)) if len(dc) > 0 else 0.0)
        if caps:
            order = np.argsort(cycles)
            proc_map[bid] = (
                cell_id,
                list(np.array(cycles)[order]),
                list(np.array(caps)[order]),
                hardcoded_removed_indices,
                median_removed_indices,
            )
    return proc_map

def load_processed_trajectories() -> Dict[int, Tuple[str, List[int], List[float], List[int], List[int]]]:
    return load_processed_trajectories_from(PROCESSED_DIR)


def plot_comparison(bid: int,
                    raw_cycles: List[int], raw_caps: List[float],
                    cell_id: str,
                    proc_cycles: List[int], proc_caps: List[float],
                    hardcoded_removed: List[int], median_removed: List[int]) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    # Raw
    ax1.plot(raw_cycles, raw_caps, color='tab:blue', linewidth=1.5)
    # Overlay removed outliers on raw subplot
    if hardcoded_removed:
        # Map 1-based indices into x positions (observed cycles start at 1)
        hc_x = [idx for idx in hardcoded_removed if 1 <= idx <= len(raw_cycles)]
        hc_y = [raw_caps[idx - 1] for idx in hc_x]
        ax1.scatter(hc_x, hc_y, color='red', s=28, marker='x', label='Hard-coded outlier')
    if median_removed:
        md_x = [idx for idx in median_removed if 1 <= idx <= len(raw_cycles)]
        md_y = [raw_caps[idx - 1] for idx in md_x]
        # Make median-filtered points more obvious: filled red circles, larger size
        ax1.scatter(md_x, md_y, color='red', s=40, marker='o', label='Median-window outlier')
    title_before = f'Battery {bid} - Raw (Before)'
    # Add legend with counts if any
    labels = []
    if hardcoded_removed:
        labels.append(f'hard={len(hardcoded_removed)}')
    if median_removed:
        labels.append(f'median={len(median_removed)}')
    if labels:
        title_before += f"  [" + ", ".join(labels) + "]"
    ax1.set_title(title_before)
    ax1.set_xlabel('Observed Cycle Index')
    ax1.set_ylabel('Discharge Capacity (Ah)')
    ax1.grid(True, alpha=0.3)
    if hardcoded_removed or median_removed:
        ax1.legend(loc='best', fontsize=8)
        # Also print the exact outlier indices as text inside the subplot
        info_lines = []
        if hardcoded_removed:
            info_lines.append(f"Hard-coded removed: {sorted(hardcoded_removed)}")
        if median_removed:
            info_lines.append(f"Median removed: {sorted(median_removed)}")
        if info_lines:
            ax1.text(
                0.02,
                0.98,
                "\n".join(info_lines),
                transform=ax1.transAxes,
                va='top',
                ha='left',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='gray')
            )
    # Processed
    ax2.plot(proc_cycles, proc_caps, color='tab:green', linewidth=1.5)
    ax2.set_title(f'{cell_id} - Processed (After)')
    ax2.set_xlabel('Processed Cycle Index')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / f"battery_{bid}_capacity_comparison.png"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_three_way_comparison(
    bid: int,
    raw_cycles: List[int], raw_caps: List[float],
    diag_cell_id: str,
    diag_cycles: List[int], diag_caps: List[float],
    comp_cell_id: str,
    comp_cycles: List[int], comp_caps: List[float],
    comp_hardcoded_removed: List[int], comp_median_removed: List[int],
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)
    ax1, ax2, ax3 = axes
    # Raw
    ax1.plot(raw_cycles, raw_caps, color='tab:blue', linewidth=1.5)
    # Overlay removed outliers from complete processing on raw subplot
    if comp_hardcoded_removed:
        hc_x = [idx for idx in comp_hardcoded_removed if 1 <= idx <= len(raw_cycles)]
        hc_y = [raw_caps[idx - 1] for idx in hc_x]
        ax1.scatter(hc_x, hc_y, color='red', s=28, marker='x', label='Hard-coded outlier')
    if comp_median_removed:
        md_x = [idx for idx in comp_median_removed if 1 <= idx <= len(raw_cycles)]
        md_y = [raw_caps[idx - 1] for idx in md_x]
        ax1.scatter(md_x, md_y, color='red', s=40, marker='o', label='Median-window outlier')
    title_before = f'Battery {bid} - Raw (Before)'
    labels = []
    if comp_hardcoded_removed:
        labels.append(f'hard={len(comp_hardcoded_removed)}')
    if comp_median_removed:
        labels.append(f'median={len(comp_median_removed)}')
    if labels:
        title_before += f"  [" + ", ".join(labels) + "]"
    ax1.set_title(title_before)
    ax1.set_xlabel('Observed Cycle Index')
    ax1.set_ylabel('Discharge Capacity (Ah)')
    ax1.grid(True, alpha=0.3)
    if comp_hardcoded_removed or comp_median_removed:
        ax1.legend(loc='best', fontsize=8)
    # Diag-only processed
    ax2.plot(diag_cycles, diag_caps, color='tab:green', linewidth=1.5)
    ax2.set_title(f'{diag_cell_id} - Processed (Diag-only)')
    ax2.set_xlabel('Processed Cycle Index')
    ax2.grid(True, alpha=0.3)
    # Complete processed
    ax3.plot(comp_cycles, comp_caps, color='tab:purple', linewidth=1.5)
    ax3.set_title(f'{comp_cell_id} - Processed (Complete)')
    ax3.set_xlabel('Processed Cycle Index')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / f"battery_{bid}_diag_complete_capacity_comparison.png"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_special_two_way(
    bid: int,
    raw_cycles: List[int], raw_caps: List[float],
    proc_label: str,
    proc_cycles: List[int], proc_caps: List[float],
    hardcoded_removed: List[int] = None,
) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    ax1.plot(raw_cycles, raw_caps, color='tab:blue', linewidth=1.5)
    # Mark hard-coded removed outliers on raw subplot if provided
    if hardcoded_removed:
        hc_x = [idx for idx in hardcoded_removed if 1 <= idx <= len(raw_cycles)]
        hc_y = [raw_caps[idx - 1] for idx in hc_x]
        ax1.scatter(hc_x, hc_y, color='red', s=28, marker='x', label='Hard-coded outlier')
    ax1.set_title(f'Battery {bid} - Raw (Before)')
    ax1.set_xlabel('Observed Cycle Index')
    ax1.set_ylabel('Discharge Capacity (Ah)')
    ax1.grid(True, alpha=0.3)
    if hardcoded_removed:
        ax1.legend(loc='best', fontsize=8)
    ax2.plot(proc_cycles, proc_caps, color='tab:green', linewidth=1.5)
    ax2.set_title(proc_label)
    ax2.set_xlabel('Processed Cycle Index')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / f"battery_{bid}_raw_vs_processed.png"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    if not RAW_DIR.exists() or not PROCESSED_DIR.exists() or not PROCESSED_COMPLETE_DIR.exists():
        print("❌ Required input directories not found.")
        return
    ensure_out_dir()

    print("📁 Loading raw and processed trajectories...")
    raw_map = load_raw_trajectories()
    diag_map = load_processed_trajectories_from(PROCESSED_DIR)
    comp_map = load_processed_trajectories_from(PROCESSED_COMPLETE_DIR)

    all_ids = sorted(set(raw_map.keys()))
    if not all_ids:
        print("❌ No raw batteries found.")
        return

    print(f"🖼️ Generating comparison plots for {len(all_ids)} batteries...")
    for bid in tqdm(all_ids, desc='Plotting comparisons'):
        raw_cycles, raw_caps = raw_map.get(bid, ([], []))
        try:
            if bid in {73, 74, 75}:
                # Prefer diag-only processed, fallback to complete, else duplicate raw
                if bid in diag_map:
                    cell_id, proc_cycles, proc_caps, _, _ = diag_map[bid]
                    label = f'{cell_id} - Processed (Diag-only)'
                elif bid in comp_map:
                    cell_id, proc_cycles, proc_caps, _, _ = comp_map[bid]
                    label = f'{cell_id} - Processed (Complete)'
                else:
                    proc_cycles, proc_caps = raw_cycles, raw_caps
                    label = f'Battery {bid} - Processed (Not Available, showing Raw)'
                # Use hard-coded removed indices from the complete set when available
                hc_removed = []
                if bid in comp_map:
                    _, _, _, hc_removed, _ = comp_map[bid]
                plot_special_two_way(bid, raw_cycles, raw_caps, label, proc_cycles, proc_caps, hc_removed)
            else:
                if bid not in diag_map or bid not in comp_map:
                    # If any processed variant missing, skip this battery gracefully
                    missing = []
                    if bid not in diag_map:
                        missing.append('diag-only')
                    if bid not in comp_map:
                        missing.append('complete')
                    print(f"   ⚠️ Skipping battery {bid}: missing processed variant(s): {', '.join(missing)}")
                    continue
                diag_cell_id, diag_cycles, diag_caps, _, _ = diag_map[bid]
                comp_cell_id, comp_cycles, comp_caps, hardcoded_removed, median_removed = comp_map[bid]
                plot_three_way_comparison(
                    bid,
                    raw_cycles, raw_caps,
                    diag_cell_id, diag_cycles, diag_caps,
                    comp_cell_id, comp_cycles, comp_caps,
                    hardcoded_removed, median_removed,
                )
        except Exception as e:
            print(f"   ⚠️ Failed plotting battery {bid}: {e}")
            continue
    print(f"✅ Done. Comparison plots saved to {OUT_DIR}")


if __name__ == '__main__':
    main()



