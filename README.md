# ML Model Drift Detection and Performance Monitoring System

A complete end-to-end Machine Learning drift detection pipeline based on a research paper implementation. It tracks statistical distribution drift across training and production data using custom implementations of Population Stability Index (PSI), Kullback-Leibler (KL) Divergence, and Kolmogorov-Smirnov (KS) Test.

## Features

- **Custom Metrics**: Pure NumPy/SciPy implementations of drift metrics.
- **Automated Logging**: SQLite database to historically log and monitor analysis runs.
- **Rich Dashboard**: Streamlit and Plotly dashboard detailing feature drift and monitoring.
- **Thresholding Strategy**:
  - `HIGH RISK`: PSI > 0.2
  - `MONITOR`: PSI > 0.1 or KS P-Value < 0.05
  - `STABLE`: otherwise

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   - The initial run has already provided `data/telco_churn.csv`.

3. **Train the baseline model:**
   Before you can monitor for drift, you must train the baseline model:
   ```bash
   python train_model.py
   ```
   *This saves the RandomForest model and baseline statistics to the `baseline/` directory.*

4. **Run the Dashboard:**
   ```bash
   streamlit run dashboard.py
   ```
   *Upload a CSV file resembling the telco data (could be modified or synthesized) to see drift detection in action.*
