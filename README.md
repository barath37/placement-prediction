# Student Placement and Salary Analytics Pipeline

## Overview
This repository contains a comprehensive, end-to-end machine learning pipeline designed to analyze student academic and extracurricular data. The project predicts placement outcomes, categorizes student profiles, extracts actionable employability rules, and estimates final salary packages using a combination of supervised, unsupervised, and deep learning approaches.

## Key Features

* **Classification (Placement Prediction):** Predicts whether a student will secure a placement using Random Forest, XGBoost, and K-Nearest Neighbors (KNN).
* **Deep Learning:** Utilizes a Multi-Layer Perceptron (MLP) neural network with advanced evaluation metrics (ROC-AUC, Log Loss) to capture complex, non-linear skill relationships.
* **Clustering (Student Personas):** Groups students into distinct profiles using K-Means and Hierarchical Clustering, while utilizing DBSCAN for anomaly detection to identify students requiring specialized intervention.
* **Association Rule Mining (Skill Basket Analysis):** Applies the Apriori algorithm to discover highly actionable "Golden Rules" (e.g., High CGPA + Many Projects -> Placed) that serve as logic gates for targeted curriculum generation.
* **Regression (Salary Prediction):** Estimates the continuous salary package (LPA) of placed students using Random Forest Regressor and XGBoost Regressor.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Rule Mining:** Mlxtend
* **Data Visualization:** Matplotlib, Seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/placement-prediction.git](https://github.com/yourusername/placement-prediction.git)
