# SDU Battery Data Preprocessor

A comprehensive battery data preprocessing tool designed to clean and standardize battery charge/discharge cycle data from CSV files. This preprocessor implements a sophisticated multi-stage filtering approach to remove outliers while preserving the integrity of normal battery cycling data.

## 🚀 Features

### Multi-Stage Outlier Detection
- **Diagnostic Cycle Replacement**: Identifies and replaces diagnostic cycles (mean negative current ≈ -0.48A) with nearest normal cycle capacities
- **Hard-coded Rules**: Applies specific, manually defined outlier removal rules for known problematic batteries
- **Median-Window Filtering**: Uses 10-cycle non-overlapping windows with conservative thresholds to detect statistical outliers

### Data Processing Pipeline
- **Capacity Calculation**: Computes charge/discharge capacities using proven algorithms
- **Cycle Organization**: Reorganizes cycle indices for consistency
- **Quality Filtering**: Removes cycles with discharge capacity < 0.1 Ah

### Comprehensive Statistics & Visualization
- Tracks all filtering operations with detailed statistics
- Generates side-by-side comparison plots showing before/after preprocessing
- Annotates outlier removal points for visual validation

## 📁 Project Structure

```
├── process_scripts/
│   └── preprocess_SDU.py              # Main preprocessor
├── process_primary_use_phase.py       # Processing script
├── plot_capacity_per_battery_comparison.py  # Visualization script
├── plots/comparison_capacity_per_battery/   # Generated comparison plots
│   ├── battery_1_capacity_comparison.png
│   ├── battery_2_capacity_comparison.png
│   └── ... (85 total plots)
└── README.md                          # This file
```

## 🔧 Technical Details

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

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scipy numba tqdm matplotlib seaborn
```

### Basic Usage
```python
from process_scripts.preprocess_SDU import SDUPreprocessor

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
# Run preprocessing
python process_primary_use_phase.py

# Generate comparison plots
python plot_capacity_per_battery_comparison.py
```

## 📊 Input/Output Format

### Input Data Format
CSV files with columns:
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

## 📈 Visualization

The comparison plots show:
- **Left**: Raw capacity trajectories with outlier annotations
  - Red 'x': Hard-coded outliers
  - Red circles: Median-filtered outliers
- **Right**: Processed capacity trajectories
- **Text annotations**: Exact cycle indices of removed outliers

## ⚙️ Configuration

### Skipped Batteries
Batteries 73, 74, and 75 are automatically skipped (no testing cycles).

### Battery Parameters
- **Form factor**: Cylindrical
- **Anode**: Graphite
- **Cathode**: NMC_532
- **Voltage limits**: 3.0V - 4.2V
- **SOC interval**: [0, 1]
- **Nominal capacity**: 2.4 Ah (primary use phase)

## 🔍 Quality Assurance

### Validation Features
- Comprehensive statistics tracking
- Visual comparison plots with outlier annotations
- Detailed logging of all filtering operations
- Error handling for malformed data

### Performance Optimizations
- JIT compilation for critical functions
- Memory-efficient file processing
- Progress tracking for large datasets

## 📋 Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy.signal`: Median filtering
- `numba`: JIT compilation for performance
- `tqdm`: Progress bars
- `matplotlib`: Plotting
- `seaborn`: Enhanced plotting styles

## 🤝 Contributing

When modifying the preprocessor:
1. Maintain the multi-stage filtering approach
2. Update hard-coded rules with proper documentation
3. Test with sample data before deployment
4. Update statistics tracking for new filtering methods

## 📄 License

This project is part of the BatteryLife project and follows the same licensing terms.

---

**Note**: This preprocessor is specifically designed for SDU battery datasets and implements sophisticated outlier detection while preserving data integrity. The visualization tools provide comprehensive validation of the preprocessing effectiveness.

