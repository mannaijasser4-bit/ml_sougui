# Sougui Data Science ML Pipeline

This project implements a full Machine Learning pipeline for Sougui's Data Warehouse, focusing on customer segmentation and classification using B2C and B2B data.

## Features

- **Data Connection**: Connect to SQL Server using pyodbc.
- **Feature Engineering**: Calculate RFM (Recency, Frequency, Monetary) metrics and scale features.
- **Clustering**: K-Means and Hierarchical Clustering with evaluation.
- **Classification**: Random Forest and Logistic Regression with hyperparameter tuning.
- **Visualization**: Scatter plots, confusion matrices, ROC-AUC, feature importance.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure SQL Server connection in the script.
3. Run the pipeline: `python src/main.py`

## Structure

- `src/`: Source code
- `notebooks/`: Jupyter notebooks for exploration
- `data/`: Data files
- `models/`: Saved models
- `reports/`: Output reports and plots