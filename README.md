Markdown# EPICS Carbon Platform: Agro-Carbon Engine

> An integrated machine learning and biophysical platform designed to optimize crop selection for farmers based on soil parameters while calculating potential carbon sequestration yields and carbon credit revenue.

---

## 🏗️ System Architecture & Data Flow

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                      FARMER INPUTS                          │
 │  (Nitrogen, Phosphorus, Potassium, Temp, Humidity, pH, Rain)│
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                  MODEL A: Crop Viability                    │
 │             (RandomForestClassifier - 99.55%)               │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼ [Outputs Top Viable Species & Probabilities]
 ┌─────────────────────────────────────────────────────────────┐
 │               MODEL B: Carbon Sequestration Engine          │
 │  (Biomass Accumulation Rates × Carbon Fraction × 44/12)     │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                  OUTPUT & DASHBOARD RESULT                  │
 │   • High-Yield Crop Recommendations                         │
 │   • Estimated Carbon Capture Yield (tCO2e/ha/yr)            │
 │   • Projected Annual Carbon Credit Revenue ($)              │
 └──────────────────────────────┴──────────────────────────────┘


## 📌 Project Overview

This repository hosts **Model A** and **Model B**, the core predictive engines of the **EPICS Carbon Platform**:

* **Model A (Crop Viability Engine):** A high-accuracy classification model that analyzes 7 key soil and environmental parameters to recommend the most suitable crop species for a given plot of land.
* **Model B (Carbon Sequestration Engine):** A biophysical calculation pipeline that takes the viable species output from Model A and estimates annual carbon dioxide equivalent ($\text{tCO}_2\text{e/ha/yr}$) sequestration and financial yields.

---

## ⚙️ Model A: Crop Viability Classifier

Model A uses a **Random Forest Classifier** trained on multi-variable agricultural data to predict crop viability probabilities.

### Input Parameters (7 Features)
* **Soil Nutrients:** Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$)
* **Climate Conditions:** Temperature (°C), Relative Humidity (%), Rainfall (mm)
* **Soil Chemistry:** pH level

### Performance Metrics
* **Accuracy:** **99.55%** on unseen test data
* **Artifact Export:** Saved to `exported_models/survival_model.pkl` for API integration

---


##📂 Project StructurePlaintext

epics-carbon-platform/
│
├── data/
│   └── raw/
│       └── Crop_recommendation.csv
│
├── exported_models/
│   └── survival_model.pkl      # Saved Model A binary
│
├── notebooks/
│   └── Model_A_Exploration.ipynb
│
├── src/
│   └── survival_Classifier.py  # Model A training & export script
│
├── requirements.txt
└── README.md


##🚀 Getting Started
### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

###2. Run Model A Pipeline
```bash
python src/survival_Classifier.py


##👨‍💻 Authors & EPICS Project
Developed as part of the EPICS Carbon Platform project to empower sustainable agriculture and carbon credit monetization for farmers.
---

