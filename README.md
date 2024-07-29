# Logistic Regression Model for Breast Cancer Detection

## Introduction
In this project, we train a logistic regression model to predict whether a given cell is malignant or not based on its measurements. This serves as an introduction to logistic regression without delving into the mathematical intuition behind the model.

The dataset used is the Breast Cancer dataset, which contains detailed measurements of cells along with their diagnosis (malignant or benign). The goal is to create a model that can accurately predict the diagnosis based on these measurements.

## Dataset
The dataset used in this project is obtained from Kaggle. You can access the dataset [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

## Project Steps

### 1. Data Gathering and Cleaning
- Imported the dataset and explored its contents.
- Cleaned the data by removing columns with NAs and the ID column, and converted the target variable into binary format (0 for benign, 1 for malignant).

### 2. Exploratory Data Analysis (EDA)
- Used visualizations to understand the distribution and relationships of the features.

### 3. Model Training
- Split the data into training and test sets.
- Trained a logistic regression model on the training set.

### 4. Model Evaluation
- Evaluated the model's performance using accuracy, precision, recall, and F1-score.
- Achieved a final accuracy of 0.98 on the test set.

### 5. Conclusion
- The trained model can accurately predict whether a cell is malignant or benign based on its measurements.
- This model can be integrated into a web application or a backend server to assist doctors in diagnosing breast cancer.

## Repository Structure
- `Logistic Regression project-checkpoint.ipynb`: The Jupyter notebook containing the entire analysis and model training process.
- `data.csv`: The dataset used for training and evaluating the model.

## Dependencies
- pandas
- seaborn
- matplotlib
- scikit-learn

## Acknowledgements
- Kaggle for providing the dataset.
- The authors of the Python libraries used in this project.
- Alejandro AO (My mentor)
