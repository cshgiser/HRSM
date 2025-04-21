# HRSM: High-Resolution Soil Moisture Mapping

A repository for high-resolution mapping of surface and rootzone soil moisture over U.S. croplands. This framework integrates multi-source remote sensing data, machine learning, and the **Layered Green and Ampt Infiltration with Redistribution (LGAR)** model.

## 📄 Research Paper

**Title:**  
*High-resolution surface and rootzone soil moisture over US cropland: A novel framework assimilating multi-source remote sensing data, machine learning, and the Layered Green and Ampt Infiltration with Redistribution model*

**Authors:**  
Shuohao Cai, *et al.*

**Contact:**  
📧 [scai62@wisc.edu](mailto:scai62@wisc.edu)

---

## 📁 Repository Structure

### 1. `HRSM_MAPPING_MachineLearning.ipynb`

- Google Colab notebook focused on the **Machine Learning (ML)** component of the HRSM framework.
- Utilizes data from **431 cropland sites** to train and validate ML models for soil moisture prediction.

### 2. `HRSM_MAPPING_pipeline_singlePoint.ipynb`

- Demonstrates the **full pipeline** of the HRSM soil moisture mapping framework.
- Executes the process at a single point using:
  - A **pre-trained ML model** (from the `StartFolder`)
  - The compiled **LGAR model** and **LGAR-EnKF model** (executables for Unix systems)
  - Input/output data required by both models

---

## 🧠 Core Components

- **Machine Learning Models:**  
  Used to capture the nonlinear relationship between input features and observed soil moisture.

- **LGAR Model:**  
  A physically based infiltration and redistribution model.  
  👉 Original source code: [NOAA-OWP/LGAR-C](https://github.com/NOAA-OWP/LGAR-C)

- **LGAR-EnKF Model:**  
  An enhanced version of LGAR integrated with the Ensemble Kalman Filter for improved accuracy using data assimilation.

---

## 🚀 Getting Started

To run the full mapping pipeline:
1. Clone this repository.
2. Open and configure `HRSM_MAPPING_pipeline_singlePoint.ipynb` in Google Colab.
3. Ensure access to:
   - The required ML model folder (`StartFolder`)
   - Compiled LGAR and LGAR-EnKF executables
   - All necessary input files

---

## 📌 Notes

- This repository is research-focused and intended for reproducibility and educational use.
- Future updates may include a streamlined pipeline for spatial mapping at larger scales.

---

## 📜 License

This project is shared under an open research license. For any usage beyond academic research, please contact the corresponding author.

---
