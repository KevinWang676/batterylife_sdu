# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from numba import njit
from typing import List
from pathlib import Path
from scipy.signal import medfilt

from batteryml import BatteryData, CycleData
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor


@PREPROCESSORS.register()
class SDUPreprocessor(BasePreprocessor):
    def process(self, parentdir, **kwargs) -> List[BatteryData]:
        path = Path(parentdir)
        raw_files = [Path(f) for f in path.glob('*.csv')]
        
        if not raw_files:
            print("No CSV files found in the directory")
            return 0, 0
        
        if not self.silent:
            print(f"Found {len(raw_files)} CSV files to process")

        process_batteries_num = 0
        skip_batteries_num = 0
        
        # Statistics tracking
        total_raw_cycles = 0
        total_outliers_removed = 0
        total_capacity_filtered = 0
        total_hardcoded_removed = 0
        total_median_removed = 0
        total_final_cycles = 0
        
        # Process each CSV file
        for csv_file in tqdm(raw_files, desc="Processing CSV files"):
            if not self.silent:
                print(f'Processing {csv_file.name}')
            
            # Load the CSV file
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
            
            # Group by Battery_ID to handle multiple batteries in one file
            for battery_id, battery_df in df.groupby('Battery_ID'):
                # Skip batteries 73, 74, 75 as requested
                try:
                    _bid = int(battery_id)
                except Exception:
                    _bid = battery_id
                if _bid in {73, 74, 75}:
                    if not self.silent:
                        print(f"Skipping Battery_{battery_id} per configuration (no testing cycles)")
                    continue
                cell_name = f"Battery_{battery_id}"
                
                # Check whether to skip the processed file
                whether_to_skip = self.check_processed_file(f'SDU_{cell_name}')
                if whether_to_skip == True:
                    skip_batteries_num += 1
                    continue
                
                if not self.silent:
                    print(f'Processing battery {cell_name}')
                
                # Prepare data in the same format as CALCE
                # Since we don't have dates, we'll use a dummy date
                battery_df = battery_df.copy()
                battery_df['date'] = '2023-01-01'  # Dummy date for consistency
                
                # Sort by Test_Time(s) - equivalent to CALCE's date+time sorting
                battery_df = battery_df.sort_values(['Test_Time(s)'])
                
                # Organize cycle index using the same function as CALCE
                battery_df['Cycle_Index'] = organize_cycle_index(battery_df['Cycle_Index'].values)
                
                # Extract required columns
                columns_to_keep = [
                    'date', 'Cycle_Index', 'Test_Time(s)', 'Current(A)', 'Voltage(V)'
                ]
                processed_df = battery_df[columns_to_keep]
                
                clean_cycles, cycles = [], []
                for cycle_index, (_, cycle_df) in enumerate(processed_df.groupby(['date', 'Cycle_Index'])):
                    I = cycle_df['Current(A)'].values  # noqa
                    t = cycle_df['Test_Time(s)'].values
                    V = cycle_df['Voltage(V)'].values
                    
                    # Calculate charge and discharge capacities using the same function as CALCE
                    Qd = calc_Q(I, t, is_charge=False)
                    Qc = calc_Q(I, t, is_charge=True)
                    
                    cycles.append(CycleData(
                        cycle_number=cycle_index,
                        voltage_in_V=V.tolist(),
                        current_in_A=I.tolist(),
                        time_in_s=t.tolist(),
                        charge_capacity_in_Ah=Qc.tolist(),
                        discharge_capacity_in_Ah=Qd.tolist()
                    ))
                
                # Clean the cycles using the same logic as CALCE
                Qd = []
                for cycle_data in cycles:
                    Qd.append(max(cycle_data.discharge_capacity_in_Ah))

                # Track raw cycles count
                raw_cycles_count = len(Qd)
                total_raw_cycles += raw_cycles_count
                
                if len(Qd) == 0:
                    print(f"No valid cycles found for battery {cell_name}")
                    continue
                
                # Identify diagnostic cycles by current profile:
                # For each cycle, compute the mean of negative currents.
                # If the mean negative current is close to -0.48 A, mark as diagnostic.
                target_neg_current = -0.48
                tolerance_in_A = 0.03  # allow small deviation around -0.48 A

                diagnostic_mask = np.zeros(len(cycles), dtype=bool)
                for i, cyc in enumerate(cycles):
                    I_arr = np.asarray(cyc.current_in_A, dtype=float)
                    neg_I = I_arr[I_arr < 0]
                    if neg_I.size == 0:
                        continue
                    mean_neg_I = float(np.mean(neg_I))
                    if abs(mean_neg_I - target_neg_current) <= tolerance_in_A:
                        diagnostic_mask[i] = True

                # Replace discharge capacity of diagnostic cycles with nearest normal cycle
                diagnostic_replaced = 0
                for i, is_diag in enumerate(diagnostic_mask):
                    if not is_diag:
                        continue
                    neighbor_idx = None
                    # search outward for nearest non-diagnostic cycle
                    for off in range(1, len(cycles)):
                        left = i - off
                        if left >= 0 and not diagnostic_mask[left]:
                            neighbor_idx = left
                            break
                        right = i + off
                        if right < len(cycles) and not diagnostic_mask[right]:
                            neighbor_idx = right
                            break
                    if neighbor_idx is not None:
                        cycles[i].discharge_capacity_in_Ah = list(cycles[neighbor_idx].discharge_capacity_in_Ah)
                        diagnostic_replaced += 1
                    # if no neighbor found, leave as is

                # Recompute Qd after potential replacements
                Qd = []
                for cycle_data in cycles:
                    Qd.append(max(cycle_data.discharge_capacity_in_Ah) if len(cycle_data.discharge_capacity_in_Ah) > 0 else 0.0)

                # Apply hard-coded outlier removal rules per battery
                hardcoded_remove_indices = set()
                if isinstance(_bid, int):
                    # Use 1-based cycle number for rule expressions (i+1)
                    if _bid == 2:
                        # Remove cycles with discharge capacity < 1.7 Ah
                        for i, q in enumerate(Qd):
                            if q < 1.7:
                                hardcoded_remove_indices.add(i)
                    elif _bid == 11:
                        # Remove cycles with cycle index > 425 and discharge capacity > 2.21 Ah
                        for i, q in enumerate(Qd):
                            if (i + 1) > 425 and q > 2.21:
                                hardcoded_remove_indices.add(i)
                    elif _bid == 17:
                        # Remove cycles with 200 <= index <= 250 and discharge capacity < 2.4 Ah
                        for i, q in enumerate(Qd):
                            ci = i + 1
                            if 200 <= ci <= 250 and q < 2.4:
                                hardcoded_remove_indices.add(i)
                    elif _bid == 21:
                        # Remove cycles with 350 <= index <= 395 and discharge capacity < 2.2 Ah
                        for i, q in enumerate(Qd):
                            ci = i + 1
                            if 350 <= ci <= 395 and q < 2.2:
                                hardcoded_remove_indices.add(i)
                    elif _bid == 46:
                        # Remove the single cycle with lowest discharge capacity in 200 <= index <= 240
                        min_q = None
                        min_i = None
                        for i, q in enumerate(Qd):
                            ci = i + 1
                            if 200 <= ci <= 240:
                                if (min_q is None) or (q < min_q):
                                    min_q = q
                                    min_i = i
                        if min_i is not None:
                            hardcoded_remove_indices.add(min_i)
                    elif _bid == 50:
                        # Remove cycles in 300..400 with q < 2.0, and in 900..1000 with q < 1.85
                        for i, q in enumerate(Qd):
                            ci = i + 1
                            if (300 <= ci <= 400 and q < 2.0) or (900 <= ci <= 1000 and q < 1.85):
                                hardcoded_remove_indices.add(i)

                # Median-window (10-cycle) outlier removal AFTER excluding hard-coded removed cycles
                # Less aggressive: non-overlapping windows; remove at most 1 clear outlier per 10-cycle block
                median_remove_indices = set()
                median_removed_indices_list = []  # candidates flagged (1-based, original indexing)
                n_cycles = len(Qd)
                indices_to_consider = [i for i in range(n_cycles) if i not in hardcoded_remove_indices]
                m_cycles = len(indices_to_consider)
                if m_cycles >= 3:
                    for start in range(0, m_cycles, 10):  # non-overlapping blocks of 10
                        end = min(start + 10, m_cycles)
                        window_indices_comp = list(range(start, end))
                        # Only evaluate full 10-cycle windows to be conservative
                        if len(window_indices_comp) < 10:
                            continue
                        window_values = [Qd[indices_to_consider[i_comp]] for i_comp in window_indices_comp]
                        m = float(np.median(window_values))
                        abs_devs = [abs(v - m) for v in window_values]
                        mad = float(np.median(abs_devs))
                        # Less aggressive thresholds to avoid removing normal cycles
                        threshold_abs = max(3.6 * mad, 0.14)
                        threshold_rel = 0.062 * max(m, 1e-6)
                        top_idx_local = int(np.argmax(abs_devs))
                        top_dev = abs_devs[top_idx_local]
                        # Dominance vs second highest deviation
                        sorted_devs = sorted(abs_devs, reverse=True)
                        second_dev = sorted_devs[1] if len(sorted_devs) > 1 else 0.0
                        dominance_ok = (second_dev == 0.0) or ((top_dev / max(second_dev, 1e-6)) >= 2.1)
                        # Modified z-score fallback (more conservative)
                        mod_z_ok = False
                        if mad > 0:
                            mod_z = 0.6745 * (top_dev / mad)
                            mod_z_ok = mod_z >= 3.6
                        if ((top_dev > threshold_abs) and (top_dev > threshold_rel) and dominance_ok) or mod_z_ok:
                            global_idx = indices_to_consider[window_indices_comp[top_idx_local]]
                            median_remove_indices.add(global_idx)
                            median_removed_indices_list.append(global_idx + 1)

                # Count filtering statistics
                capacity_filtered_count = 0
                hardcoded_removed_count = 0
                median_removed_count = 0
                median_removed_actual_indices_list = []
                final_cycles_count = 0
                
                clean_cycles, index = [], 0
                for i in range(len(cycles)):
                    # Skip hard-coded removed cycles
                    if i in hardcoded_remove_indices:
                        hardcoded_removed_count += 1
                        continue
                    # Skip median-window outliers
                    if i in median_remove_indices:
                        median_removed_count += 1
                        median_removed_actual_indices_list.append(i + 1)
                        continue
                    # Default capacity-based filter
                    if Qd[i] > 0.1:
                        index += 1
                        cycles[i].cycle_number = index
                        clean_cycles.append(cycles[i])
                        final_cycles_count += 1
                    else:
                        capacity_filtered_count += 1
                
                # Update global statistics
                total_outliers_removed += diagnostic_replaced
                total_capacity_filtered += capacity_filtered_count
                total_hardcoded_removed += hardcoded_removed_count
                total_median_removed += median_removed_count
                total_final_cycles += final_cycles_count
                
                # Print battery-specific statistics
                if not self.silent:
                    print(f"   Battery {cell_name} stats: {raw_cycles_count} raw → {raw_cycles_count - capacity_filtered_count} final")
                    if diagnostic_replaced > 0:
                        print(f"     - Diagnostic cycles replaced (mean neg. I ≈ -0.48 A): {diagnostic_replaced}")
                    if capacity_filtered_count > 0:
                        print(f"     - Low capacity filtered: {capacity_filtered_count}")
                    if hardcoded_removed_count > 0:
                        print(f"     - Hard-coded outliers removed: {hardcoded_removed_count}")
                    if median_removed_count > 0:
                        print(f"     - Median-window outliers removed: {median_removed_count} at cycles {sorted(set(median_removed_actual_indices_list))}")
                
                if len(clean_cycles) == 0:
                    print(f"No clean cycles found for battery {cell_name}")
                    continue
                
                # Estimate nominal capacity from the first few cycles
                C = 2.4 # primary use phase: 2.4; second life phase: 1.92
                
                # Set default battery parameters
                soc_interval = [0, 1]
                
                # Prepare 1-based indices for persistence
                hardcoded_removed_indices_list = sorted([i + 1 for i in hardcoded_remove_indices])
                median_removed_indices_persist = sorted(set(median_removed_actual_indices_list))
                
                battery = BatteryData(
                    cell_id=f'SDU_{cell_name}',
                    form_factor='cylindrical',
                    anode_material='graphite',
                    cathode_material='NMC_532',
                    cycle_data=clean_cycles,
                    nominal_capacity_in_Ah=C,
                    max_voltage_limit_in_V=4.2,
                    min_voltage_limit_in_V=3,
                    SOC_interval=soc_interval,
                    hardcoded_removed_indices=hardcoded_removed_indices_list,
                    median_removed_indices=median_removed_indices_persist
                )
                
                self.dump_single_file(battery)
                process_batteries_num += 1
                
                if not self.silent:
                    tqdm.write(f'File: {battery.cell_id} dumped to pkl file')
        
        # Print final statistics summary
        if not self.silent:
            print(f"\n📊 PREPROCESSING STATISTICS SUMMARY:")
            print(f"=" * 50)
            print(f"Total raw cycles processed: {total_raw_cycles:,}")
            print(f"Diagnostic cycles replaced (mean neg. I ≈ -0.48 A): {total_outliers_removed:,} ({100*total_outliers_removed/total_raw_cycles:.1f}%)")
            print(f"Low capacity filtered (<0.1 Ah): {total_capacity_filtered:,} ({100*total_capacity_filtered/total_raw_cycles:.1f}%)")
            print(f"Hard-coded outliers removed: {total_hardcoded_removed:,} ({100*total_hardcoded_removed/total_raw_cycles:.1f}%)")
            print(f"Median-window outliers removed: {total_median_removed:,} ({100*total_median_removed/total_raw_cycles:.1f}%)")
            print(f"Final clean cycles retained: {total_final_cycles:,} ({100*total_final_cycles/total_raw_cycles:.1f}%)")
            print(f"Total data reduction: {total_raw_cycles - total_final_cycles:,} cycles ({100*(total_raw_cycles - total_final_cycles)/total_raw_cycles:.1f}%)")
        
        return process_batteries_num, skip_batteries_num


@njit
def calc_Q(I, t, is_charge):  # noqa
    """
    Calculate charge/discharge capacity - same function as CALCE preprocessor
    """
    Q = np.zeros_like(I)
    for i in range(1, len(I)):
        if is_charge and I[i] > 0:
            Q[i] = Q[i-1] + I[i] * (t[i] - t[i-1]) / 3600
        elif not is_charge and I[i] < 0:
            Q[i] = Q[i-1] - I[i] * (t[i] - t[i-1]) / 3600
        else:
            Q[i] = Q[i-1]
    return Q


@njit
def organize_cycle_index(cycle_index):
    """
    Organize cycle indices - same function as CALCE preprocessor
    """
    current_cycle, prev_value = cycle_index[0], cycle_index[0]
    for i in range(1, len(cycle_index)):
        if cycle_index[i] != prev_value:
            current_cycle += 1
            prev_value = cycle_index[i]
        cycle_index[i] = current_cycle
    return cycle_index 