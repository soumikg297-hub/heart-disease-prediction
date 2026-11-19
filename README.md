# Heart Disease Prediction Using Machine Learning

## Project objective
Build an educational binary-classification prototype that uses clinical attributes from the Cleveland subset of the UCI Heart Disease dataset to classify records into dataset labels: no disease (0) or disease present (1–4 collapsed to 1).

**Important:** This is not a medical diagnostic device. It must not be used to diagnose, rule out disease, estimate an individual's real-world risk, or guide treatment. The dataset is small and historical, and test metrics do not establish clinical validity.

## Features
- Python machine-learning pipeline using Random Forest
- Missing-value handling with median imputation
- Stratified train/test split
- Accuracy, precision, recall, F1-score and confusion matrix
- Streamlit form-based user interface

## Requirements
- Python 3.9 or newer
- Internet connection for the first dataset download

## Run on Windows (VS Code / Command Prompt)
1. Extract the ZIP and open the project folder in VS Code.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model and download the dataset:
   ```bash
   python train_model.py
   ```
5. Start the web app:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL shown in the terminal (usually http://localhost:8501).

The dataset is downloaded from the UCI Machine Learning Repository on first run. Source: https://archive.ics.uci.edu/dataset/45/heart+disease

## Dataset input fields
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.

The dataset's original target values are 0–4. This project converts them to binary labels: 0 remains 0; values 1–4 become 1.

## Suggested project report chapters
1. Introduction
2. Problem statement and objectives
3. Literature / dataset review
4. System requirements
5. Methodology and algorithm
6. System design and flowchart
7. Implementation
8. Testing and evaluation
9. Limitations and future scope
10. Conclusion and references

## Reference
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
