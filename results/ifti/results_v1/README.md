# Adversarial Training Results
## Files Description

### CSV Files
- `comprehensive_results.csv` - All evaluation results (raw data)
- `summary_table.csv` - Summary statistics per model
- `baseline_vs_defended.csv` - Baseline vs defended comparison
- `attack_specific_performance.csv` - Performance against specific attacks

### LaTeX Tables
- `table_summary.tex` - Summary table
- `table_comparison.tex` - Comparison table
- `table_attack_specific.tex` - Attack-specific table

### Reports
- `summary_report.txt` - Comprehensive text report
- `README.md` - This file

### Model Files
- Individual model JSON files (`*_evaluation.json`)
- Individual model CSV files (`*_evaluation_summary.csv`)
- Confusion matrices (`*_confusion_matrices.npz`)
- Training histories (`*_history.json`)

## Usage

### For Analysis
```python
import pandas as pd
results = pd.read_csv('comprehensive_results.csv')
```

### For LaTeX Paper
```latex
\input{results/table_summary.tex}
```
