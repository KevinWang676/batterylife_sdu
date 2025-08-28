# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from numba import njit
from typing import List
from pathlib import Path
 

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

                # Keep all cycles (no hard-coded removals, no median-window filter, no <0.1 Ah filter)
                final_cycles_count = 0
                clean_cycles, index = [], 0
                for i in range(len(cycles)):
                    index += 1
                    cycles[i].cycle_number = index
                    clean_cycles.append(cycles[i])
                    final_cycles_count += 1
                
                # Update global statistics
                total_outliers_removed += diagnostic_replaced
                total_final_cycles += final_cycles_count
                
                # Print battery-specific statistics
                if not self.silent:
                    print(f"   Battery {cell_name} stats: {raw_cycles_count} raw → {final_cycles_count} final")
                    if diagnostic_replaced > 0:
                        print(f"     - Diagnostic cycles replaced (mean neg. I ≈ -0.48 A): {diagnostic_replaced}")
                
                if len(clean_cycles) == 0:
                    print(f"No clean cycles found for battery {cell_name}")
                    continue
                
                # Estimate nominal capacity from the first few cycles
                C = 2.4 # primary use phase: 2.4; second life phase: 1.92
                
                # Set default battery parameters
                soc_interval = [0, 1]
                
                # Prepare 1-based indices for persistence (none removed)
                hardcoded_removed_indices_list = []
                median_removed_indices_persist = []
                
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
            print(f"Final clean cycles retained: {total_final_cycles:,} ({100*total_final_cycles/total_raw_cycles:.1f}%)")
        
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
