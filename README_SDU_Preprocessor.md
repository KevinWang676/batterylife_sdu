# SDU Battery Data Preprocessor

## Overview

The SDU Preprocessor (`preprocess_SDU.py`) is a comprehensive battery data preprocessing tool designed to clean and standardize battery charge/discharge cycle data from CSV files. It implements a multi-stage filtering approach to remove outliers while preserving the integrity of normal battery cycling data.

## Features

### 1. Multi-Stage Outlier Detection
- **Diagnostic Cycle Replacement**: Identifies and replaces diagnostic cycles (mean negative current ≈ -0.48A) with nearest normal cycle capacities
- **Hard-coded Rules**: Applies specific, manually defined outlier removal rules for known problematic batteries
- **Median-Window Filtering**: Uses 10-cycle non-overlapping windows with conservative thresholds to detect statistical outliers

### 2. Data Processing Pipeline
- **Capacity Calculation**: Computes charge/discharge capacities using the same algorithm as CALCE preprocessor
- **Cycle Organization**: Reorganizes cycle indices for consistency
- **Quality Filtering**: Removes cycles with discharge capacity < 0.1 Ah

### 3. Comprehensive Statistics
- Tracks all filtering operations with detailed statistics
- Reports outlier removal counts and percentages
- Provides battery-specific processing summaries

## Technical Details

### Diagnostic Cycle Detection
```python
target_neg_current = -0.48
tolerance_in_A = 0.03
# Identifies cycles with mean negative current close to -0.48A
# Replaces their discharge capacity with nearest normal cycle
```

### Hard-coded Outlier Rules
- **Battery 2**: Remove cycles with discharge capacity < 1.7 Ah
- **Battery 11**: Remove cycles > 425 with capacity > 2.21 Ah
- **Battery 17**: Remove cycles 200-250 with capacity < 2.4 Ah
- **Battery 21**: Remove cycles 350-395 with capacity < 2.2 Ah
- **Battery 46**: Remove lowest capacity cycle in range 200-240
- **Battery 50**: Remove cycles 300-400 with capacity < 2.0 Ah, and cycles 900-1000 with capacity < 1.85 Ah

### Median-Window Filtering
- **Window Size**: 10 cycles (non-overlapping)
- **Thresholds**: Conservative settings to avoid removing normal cycles
  - Absolute threshold: max(3.6 × MAD, 0.14)
  - Relative threshold: 0.062 × median(window)
  - Dominance ratio: ≥ 2.1
  - Modified z-score: ≥ 3.6
- **Output**: At most one outlier removed per 10-cycle window

## Usage

### Basic Usage
```python
from preprocess_SDU import SDUPreprocessor

# Initialize preprocessor
preprocessor = SDUPreprocessor(
    output_dir="./processed_data",
    silent=False
)

# Process CSV files
processed_num, skipped_num = preprocessor.process(
    parentdir="/path/to/csv/files"
)
```

### Command Line Usage
```bash
# Activate virtual environment
source .venv/bin/activate

# Run preprocessing
python process_primary_use_phase.py
```

### Input Data Format
The preprocessor expects CSV files with the following columns:
- `Battery_ID`: Battery identifier
- `Cycle_Index`: Cycle number
- `Test_Time(s)`: Test time in seconds
- `Current(A)`: Current in amperes
- `Voltage(V)`: Voltage in volts
- `Discharge_Capacity(Ah)`: Discharge capacity (optional, will be calculated)

### Output Format
- **Pickle files**: `SDU_Battery_{id}.pkl` containing `BatteryData` objects
- **Metadata**: Includes outlier removal indices for analysis
- **Statistics**: Comprehensive processing summaries

## Visualization

The `plot_capacity_per_battery_comparison.py` script generates side-by-side comparison plots:
- **Left**: Raw capacity trajectories with outlier annotations
- **Right**: Processed capacity trajectories
- **Annotations**: Red 'x' for hard-coded outliers, red circles for median-filtered outliers

## Configuration

### Skipped Batteries
Batteries 73, 74, and 75 are automatically skipped (no testing cycles).

### Nominal Capacity
- Primary use phase: 2.4 Ah
- Second life phase: 1.92 Ah

### Battery Parameters
- Form factor: Cylindrical
- Anode: Graphite
- Cathode: NMC_532
- Voltage limits: 3.0V - 4.2V
- SOC interval: [0, 1]

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy.signal`: Median filtering
- `numba`: JIT compilation for performance
- `tqdm`: Progress bars
- `batteryml`: Battery data structures and base classes

## Performance

- **JIT Optimization**: Critical functions (`calc_Q`, `organize_cycle_index`) use Numba JIT compilation
- **Memory Efficient**: Processes files individually to manage memory usage
- **Progress Tracking**: Real-time progress bars for large datasets

## Quality Assurance

### Validation Features
- Comprehensive statistics tracking
- Visual comparison plots
- Outlier annotation in plots
- Detailed logging of all filtering operations

### Error Handling
- Graceful handling of malformed CSV files
- Skip processing for problematic batteries
- Detailed error reporting

## File Structure

```
├── preprocess_SDU.py                    # Main preprocessor
├── process_primary_use_phase.py         # Processing script
├── plot_capacity_per_battery_comparison.py  # Visualization script
└── plots/comparison_capacity_per_battery/   # Generated plots
    ├── battery_1_capacity_comparison.png
    ├── battery_2_capacity_comparison.png
    └── ...
```

## Contributing

When modifying the preprocessor:
1. Maintain the multi-stage filtering approach
2. Update hard-coded rules with proper documentation
3. Test with sample data before deployment
4. Update statistics tracking for new filtering methods

## License

This preprocessor is part of the BatteryLife project and follows the same licensing terms.
