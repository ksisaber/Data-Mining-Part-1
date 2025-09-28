# Data Preprocessing and Exploratory Analysis

This project focuses on **exploratory data analysis (EDA)** and **data preprocessing** using a real-world climate dataset. Since real data is often noisy, large, and heterogeneous, the project applies systematic steps to clean, explore, and transform the data for further mining and machine learning tasks.

The dataset used in this project has been restricted to **Algeria** for detailed study and processing.

---

## Objectives

The project is divided into two main phases:

### 1. Exploratory Data Analysis (EDA)

* Import, visualize, and save dataset contents
* Provide a global description of the dataset
* Update or delete instances or values
* Attribute-level analysis:

  * Central tendency measures (mean, median, mode)
  * Dispersion measures (variance, standard deviation, range)
  * Outlier detection
  * Missing values and unique values count
  * Boxplots for outliers
  * Histograms for distribution visualization
  * Scatter plots for correlation detection

### 2. Data Preprocessing

Based on insights from the first phase, the dataset is cleaned and transformed to be **100% functional** and optimized for subsequent steps. Functionalities include:

* **Data Reduction**: aggregation by seasons
* **Data Integration**: merging multiple sources into a unified dataset
* **Handling Missing Values/Outliers**: with multiple strategies
* **Normalization**: Min-Max scaling and Z-score standardization
* **Discretization**: Equal Frequency / Equal Width binning
* **Redundancy Elimination**: horizontal and vertical reduction

---

## Project Structure

```
final_code.ipynb        -> Jupyter notebook with complete analysis and preprocessing
final_code.py           -> Python script version of the notebook
interface1.py           -> Streamlit interface (Step 1: EDA)
interface2.py           -> Streamlit interface (Step 2: Preprocessing)
interface.py            -> Final Streamlit interface (Step 1 + Step 2 combined)
part1.py                -> EDA scripts
part2.py                -> Preprocessing scripts
soil_dz_allprops.csv    -> Climate dataset (Algeria subset)
```

---

## Requirements

* Python 3.8+
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

---

## How to Run

### Option 1: Jupyter Notebook

Run the notebook interactively:

```bash
jupyter notebook final_code.ipynb
```

### Option 2: Python Script

Execute the full workflow:

```bash
python final_code.py
```

### Option 3: Streamlit Interface

Run one of the Streamlit apps:

* Step 1 only (EDA):

  ```bash
  streamlit run interface1.py
  ```

* Step 2 only (Preprocessing):

  ```bash
  streamlit run interface2.py
  ```

* Full application (Step 1 + Step 2):

  ```bash
  streamlit run interface.py
  ```

---

## Dataset

* **Source**: Global climate dataset (subset restricted to Algeria)
* **File**: `soil_dz_allprops.csv`
* **Attributes**: soil and climate-related properties (numeric and categorical)

---

## Outputs

* Descriptive statistics for each attribute
* Visualizations (histograms, boxplots, scatter plots, correlation matrices)
* Cleaned dataset after preprocessing
* Reduced/normalized dataset ready for mining and modeling

---

## Authors

* Project developed as part of the **Data Mining and Preprocessing coursework**.
