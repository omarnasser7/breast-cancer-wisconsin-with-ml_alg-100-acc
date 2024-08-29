# Breast Cancer Wisconsin (Diagnostic) Dataset - 100% Accuracy with ML Algorithms

## Overview
This notebook demonstrates how to achieve 100% accuracy in predicting breast cancer diagnoses using machine learning algorithms on the [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). The dataset consists of features computed from digitized images of breast mass tissue samples, with the goal of classifying tumors as benign (B) or malignant (M).

## Dataset
The dataset is available on Kaggle and contains 569 samples with 30 features. The features include:
- Mean, standard error, and worst values of metrics like radius, texture, perimeter, area, and smoothness, among others.

The target variable is `diagnosis`, where:
- `B` = Benign
- `M` = Malignant

You can access the dataset [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

## Notebook Content

### 1. Exploratory Data Analysis (EDA)
- Data Loading and Preview: The dataset is loaded and a preview of the data is provided.
- Data Cleaning: Handling missing values and preparing the data for modeling.
- Feature Correlation: Visualizations such as heatmaps are used to understand the correlation between features.
- Class Distribution: The distribution of the target variable (`diagnosis`) is displayed using visual tools like pie charts.

### 2. Data Preprocessing
- Feature Scaling: Standardizing the dataset using techniques such as StandardScaler to ensure that all features contribute equally to the model performance.
- Train-Test Split: Splitting the data into training and testing sets.

### 3. Modeling
- Machine Learning Models: Multiple machine learning algorithms are implemented and compared, including:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Random Forest
    - Decision Tree
    - Naive Bayes
- Model Evaluation: The models are evaluated using metrics such as accuracy, precision, recall, and F1-score. The classification report is generated for detailed analysis.

### 4. Results
- Model Comparison: The models are compared based on their accuracy, and the best-performing model is identified.
- Achieving 100% Accuracy: Insights and techniques used to achieve perfect accuracy on this dataset are discussed.

### 5. Conclusion
- Summary of findings and the implications of achieving 100% accuracy in a medical diagnosis context.
- Discussion on the importance of model evaluation beyond just accuracy to avoid overfitting and ensure generalization.

## Getting Started
To run this notebook:
1. Clone the notebook from Kaggle.
2. Install the necessary dependencies.
3. Run the notebook cells sequentially to reproduce the results.

You can access the notebook [here](https://www.kaggle.com/code/omarnasser7/breast-cancer-wisconsin-with-ml-alg-100-acc).

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn


## Author
Omar Nasser
