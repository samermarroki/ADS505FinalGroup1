# E-Commerce Shoppers Purchasing Intention — Data Preparation & EDA Pipeline

## Project Overview & Objective
This repository contains the data preparation, exploratory data analysis (EDA), and data splitting pipeline for the UCI **Online Shoppers Purchasing Intention Dataset** (12,330 web sessions). 

The primary business objective is to analyze visitor behavior and prepare clean, engineered, and scaled data matrices to support downstream machine learning classification models predicting whether a browsing session will result in a purchase (`Revenue`: `True`/`False`).

---

## Assigned Scope & Deliverables
This portion of the project covers four key phases:
1. **Problem Statement & Data Description**: Framing the e-commerce conversion challenge, defining the target variable, and summarizing raw feature dimensions.
2. **Exploratory Data Analysis (EDA)**: Visualizing conversion rates across visitor segments and investigating session intent metrics (`PageValues` vs. `BounceRates`).
3. **Data Pre-processing**: 
   - Missing value verification.
   - Outlier management via 99th percentile capping on high-variance duration columns.
   - Multicollinearity handling through interaction features (`BounceRates` × `ExitRates`).
   - Feature mapping (ordinal transformation for months and one-hot encoding for categorical attributes).
   - Leak-free standardization using `StandardScaler` inside a `ColumnTransformer`.
4. **Data Splitting**: Stratified 70% Train / 15% Validation / 15% Test partitioning to maintain consistent class balance (~15.5% positive conversion rate) across all subsets.

---

## Repository Structure

```text
ADS505FinalGroup1/
├── README.md                             # Documentation and handoff guide
├── Technical_Notebook-Team1.ipynb        # Consolidated final technical notebook
├── data/
│   └── online_shoppers_intention.csv     # Source dataset
└── work/                                 # Individual development notebooks
    ├── chris_feature_engineering.ipynb
    ├── Samer-Marroki.ipynb
    └── modeling_and_evaluation.ipynb
