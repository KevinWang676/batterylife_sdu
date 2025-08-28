#!/usr/bin/env python3
"""
Script to get the discharge capacity of a specific cycle for a specific battery.
"""

import pickle
from pathlib import Path

def get_cycle_capacity(battery_id: int, cycle_index: int, processed_dir: str) -> float:
    """
    Get the discharge capacity of a specific cycle for a specific battery.
    
    Args:
        battery_id: Battery ID (e.g., 51)
        cycle_index: 1-based cycle index (e.g., 156)
        processed_dir: Path to directory containing processed pickle files
    
    Returns:
        Discharge capacity in Ah, or None if not found
    """
    processed_path = Path(processed_dir)
    
    # Look for the battery's pickle file
    pkl_file = processed_path / f"SDU_Battery_{battery_id}.pkl"
    
    if not pkl_file.exists():
        print(f"❌ Pickle file not found: {pkl_file}")
        return None
    
    try:
        with open(pkl_file, 'rb') as f:
            battery_data = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")
        return None
    
    # Extract cycle data
    if isinstance(battery_data, dict):
        cycle_data = battery_data['cycle_data']
        cell_id = battery_data.get('cell_id', f'SDU_Battery_{battery_id}')
    else:
        cycle_data = battery_data.cycle_data
        cell_id = getattr(battery_data, 'cell_id', f'SDU_Battery_{battery_id}')
    
    print(f"📋 Battery: {cell_id}")
    print(f"📊 Total cycles in processed data: {len(cycle_data)}")
    
    # Find the cycle with the specified index
    target_cycle = None
    for cyc in cycle_data:
        if isinstance(cyc, dict):
            cycle_num = int(cyc['cycle_number'])
            if cycle_num == cycle_index:
                target_cycle = cyc
                break
        else:
            cycle_num = int(cyc.cycle_number)
            if cycle_num == cycle_index:
                target_cycle = cyc
                break
    
    if target_cycle is None:
        print(f"❌ Cycle {cycle_index} not found in processed data")
        print(f"Available cycles: {sorted([int(cyc['cycle_number'] if isinstance(cyc, dict) else cyc.cycle_number) for cyc in cycle_data])}")
        return None
    
    # Get discharge capacity
    if isinstance(target_cycle, dict):
        discharge_cap = target_cycle['discharge_capacity_in_Ah'] or []
    else:
        discharge_cap = getattr(target_cycle, 'discharge_capacity_in_Ah', []) or []
    
    if not discharge_cap:
        print(f"❌ No discharge capacity data found for cycle {cycle_index}")
        return None
    
    max_discharge_cap = max(discharge_cap)
    print(f"✅ Cycle {cycle_index} discharge capacity: {max_discharge_cap:.6f} Ah")
    
    return max_discharge_cap

def main():
    # Configuration
    battery_id = 51
    cycle_index = 156
    processed_dir = "/Users/kevinwang/Downloads/BatteryLife/processed_primary_use_phase_08_24"
    
    print(f"🔍 Looking for cycle {cycle_index} discharge capacity in Battery {battery_id}")
    print(f"📁 Processed data directory: {processed_dir}")
    print("-" * 60)
    
    capacity = get_cycle_capacity(battery_id, cycle_index, processed_dir)
    
    if capacity is not None:
        print("-" * 60)
        print(f"🎯 RESULT: Battery {battery_id}, Cycle {cycle_index} = {capacity:.6f} Ah")
    else:
        print("❌ Could not retrieve capacity")

if __name__ == '__main__':
    main()
