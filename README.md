# BatteryLife SDU Dataset Preprocessing

This repository contains the preprocessing pipeline for the SDU (Stanford University) battery dataset, which processes raw battery cycling data into clean, standardized formats for machine learning applications.

## Overview

The preprocessing pipeline transforms raw CSV battery cycling data into structured `BatteryData` objects with the following key features:

- **Diagnostic Cycle Detection and Replacement**: Identifies and replaces diagnostic cycles based on current profiles
- **Hard-coded Outlier Removal**: Applies battery-specific filtering rules to remove anomalous cycles
- **Standardized Data Format**: Outputs data in a consistent format compatible with the BatteryML framework

Outliers removal plots: https://docs.google.com/presentation/d/1JO6gIbKDEIW7drJZjJmtraH-c-HtvXriridjuM5T4JU/edit?usp=sharing

## Preprocessing Pipeline

### 1. Data Loading and Initial Processing

- Loads CSV files containing battery cycling data
- Groups data by `Battery_ID` to handle multiple batteries per file
- Sorts data by `Test_Time(s)` for chronological ordering
- Organizes cycle indices using the `organize_cycle_index()` function
- Extracts required columns: `date`, `Cycle_Index`, `Test_Time(s)`, `Current(A)`, `Voltage(V)`

### 2. Capacity Calculation

Uses the `calc_Q()` function to calculate charge and discharge capacities:
- **Charge Capacity**: Accumulates positive current over time
- **Discharge Capacity**: Accumulates negative current over time
- Time integration with conversion to Ampere-hours (Ah)

### 3. Diagnostic Cycle Detection and Replacement

**Detection Method:**
- Analyzes each cycle's current profile
- Computes mean negative current during discharge phases
- Identifies diagnostic cycles where mean negative current ≈ -0.48 A (±0.03 A tolerance)

**Replacement Strategy:**
- For each diagnostic cycle, finds the nearest non-diagnostic cycle
- Replaces discharge capacity with the neighbor's capacity values
- Searches outward in both directions (left and right) to find the closest normal cycle

**Special Handling:**
- **Batteries 73, 74, 75**: Diagnostic cycle replacement is **SKIPPED** for these batteries
- These batteries are processed without diagnostic cycle filtering

### 4. Hard-coded Outlier Removal

Applies battery-specific filtering rules to remove anomalous cycles based on cycle index ranges and capacity thresholds:

#### Battery-Specific Rules:

- **Battery 2**: Removes cycles > 800 with capacity < 1.7 Ah
- **Battery 11**: Removes cycles 440-500 with capacity > 2.22 Ah
- **Battery 17**: Removes cycles 200-250 with capacity < 2.4 Ah
- **Battery 48**: Removes cycles 1-100 with capacity < 2.0 Ah
- **Battery 50**: Removes cycles 200-400 with capacity < 2.0 Ah, cycles 900-1000 with capacity < 1.825 Ah
- **Battery 51**: Removes cycles 100-220 with capacity < 2.23 Ah, cycles 200-220 with capacity > 2.37 Ah
- **Battery 65**: Removes cycles > 210 with capacity > 2.11 Ah
- **Battery 73**: Removes cycles 600-650 with capacity < 2.1 Ah
- **Battery 74**: Removes cycles 650-700 with capacity < 2.24 Ah
- **Battery 75**: Removes cycles 100-150 with capacity < 2.355 Ah
- **Battery 76**: Removes cycles 900-990 with capacity < 1.982 Ah
- **Battery 80**: Removes cycles 450-540 with capacity < 2.22 Ah
- **Battery 82**: Removes cycles 490-540 with capacity < 2.245 Ah
- **Battery 83**: Removes cycles 600-700 with capacity < 2.0 Ah

#### Special Exceptions:

Certain cycles are explicitly preserved with exact capacity values:
- **Battery 48, Cycle 26**: 2.400905 Ah
- **Battery 48, Cycle 31**: 2.390913 Ah
- **Battery 50, Cycle 951**: 1.898038 Ah
- **Battery 51, Cycle 156**: 2.297773 Ah

### 5. Final Data Structure

Each processed battery is stored as a `BatteryData` object with:
- **Cell ID**: `SDU_Battery_{id}`
- **Form Factor**: Cylindrical
- **Materials**: Graphite anode, NMC_532 cathode
- **Nominal Capacity**: 2.4 Ah (primary use phase)
- **Voltage Limits**: 3.0V - 4.2V
- **SOC Interval**: [0, 1]
- **Cycle Data**: Cleaned cycle information with voltage, current, time, and capacity data
- **Removal Tracking**: Lists of hard-coded and median-filtered removed cycle indices

## Key Features

### Diagnostic Cycle Handling
- **Automatic Detection**: Uses current profile analysis to identify diagnostic cycles
- **Smart Replacement**: Replaces with nearest neighbor capacity values
- **Selective Application**: Skips diagnostic processing for batteries 73, 74, 75

### Outlier Management
- **Battery-Specific Rules**: Custom filtering criteria for each battery
- **Cycle Range Targeting**: Removes outliers in specific cycle index windows
- **Capacity Thresholds**: Uses capacity-based filtering criteria
- **Exception Handling**: Preserves specific cycles with exact capacity values

### Data Quality Assurance
- **Comprehensive Statistics**: Tracks raw cycles, diagnostic replacements, hard-coded removals, and final cycles
- **Transparency**: Maintains lists of removed cycle indices for audit purposes
- **Consistency**: Standardized output format across all batteries

## Usage

The preprocessing is implemented as a `SDUPreprocessor` class that can be used with the BatteryML framework:

```python
from batteryml.preprocess import SDUPreprocessor

preprocessor = SDUPreprocessor()
processed_batteries = preprocessor.process("path/to/raw/csv/files")
```

## Output Statistics

The preprocessing pipeline provides detailed statistics:
- Total raw cycles processed
- Number of diagnostic cycles replaced
- Number of hard-coded outliers removed
- Final clean cycles retained
- Percentage breakdowns for each category

## File Structure

- `process_scripts/process_SDU.py`: Enhanced preprocessing script (diagnostic + hard-coded filtering)
- `process_scripts/process_SDU_only_remove_diagnostic_cycles.py`: Original preprocessing script (removing and replacing diagnostic cycles only)
- `comparison_capacity_per_battery_08_28/`: Visualization plots showing before/after preprocessing
- Processed data files in pickle format for each battery

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `numba`: JIT compilation for performance
- `tqdm`: Progress tracking
- `batteryml`: Battery data framework

## License

Licensed under the MIT License. Copyright (c) Microsoft Corporation.

