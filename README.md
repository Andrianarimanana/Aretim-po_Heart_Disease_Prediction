# Aretim-po Prediction (Heart Disease) 

Welcome to the **Aretim-po Prediction or Heart Disease** project! This repository designed a project to predict heart disease risk comparing some Machine Learning models, featuring data preprocessing, model training, evaluation, and deployment-ready insights.
![Heart Disease Prediction](Image/Young-Myocarditis-Heart-Concept.webp)
## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model performance](#model)
3. [Deploiment](#deploiment)
3. [Setup Instructions](#setup-instructions)

## Overview

The porpuse of this project is a machine learning focused on assessing the risk of heart disease! 
This repository encompasses a comprehensive workflow to compare three Machine Learning models.
It includes data preprocessing, model training, performance evaluation, and deployment-ready insights to aid in predicting heart disease risk efficiently and accurately.

## Dataset

The repository includes data derived from [UCI Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) 

Dataset Details:
This dataset is multivariate, containing multiple statistical variables, and is widely used in machine learning research for heart disease prediction and analysis. It is derived from the Cleveland database and includes **14 key attributes** commonly used in published studies.

## Attributes
The dataset focuses on the following features:

- **Demographics**:
  - Age
  - Sex
- **Clinical Measurements**:
  - Chest pain type
  - Resting blood pressure
  - Serum cholesterol
  - Fasting blood sugar
  - Resting electrocardiographic results
- **Exercise Data**:
  - Maximum heart rate achieved
  - Exercise-induced angina
  - ST depression induced by exercise relative to rest ("oldpeak")
  - Slope of the ST segment during peak exercise
- **Other**:
  - Number of major vessels
  - Thalassemia

## Target Column: `num`
- The `num` column represents the predicted attribute for heart disease.
- Unique values: `[0, 1, 2, 3, 4]`, indicating 5 types of heart diseases.
   - 0 = no heart disease.
   - 1 = Mild Heart Disease types.
   - 2 = Moderate Heart Disease type.
   - 3 = Severe Heart Disease type.
   - 4 = Critical Heart Disease type.

## Notes
- The full database contains 76 attributes, but only these 14 are  used .

## Files structure in Repository:
Date/
heart_disease_uci.csv: Contains input features and labels for model training.

1. **notebook.ipynb** :contains 
      - Data preparation and data cleaning
      - EDA, feature importance analysis**
      - Model selection process and parameter tuning
2. **train.py** :selected model traianing and saving to file with pickle
3. **predict.py** :simple load model and predict , **deployed service with flask**
4. **predict-test.py** :test the flash app
5. **Pipfile.lock** : python dependancy
6. **Pipfile** : python dependancy
7. **Dockerfile** :containerization, to create docker image

## Evaluation Metric: ROC AUC Score

The recommended evaluation metric for this project is the **ROC AUC score** (Receiver Operating Characteristic - Area Under the Curve). This metric evaluates the performance of binary classification models by measuring their ability to distinguish between two classes.

### Key Concepts:
- **ROC Curve**: Plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various thresholds.
- **AUC (Area Under the Curve)**: Represents the area under the ROC curve, providing a single scalar value to summarize the model's performance.

ROC AUC is a reliable and concise metric for assessing classification models.


## Model Training

The model training process consisted of the following steps:

### 1. Data Preprocessing
- Handled missing values appropriately.
- Transformed column data types to ensure compatibility with machine learning models.

### 2. Exploratory Data Analysis (EDA)
- Conducted EDA to uncover basic patterns and relationships in the dataset.
- Identified trends and correlations among key features.

### 3. Feature Importance Analysis
- Evaluated the contribution of each variable to the prediction task.

### 4. Modeling Phase
Five algorithms were selected for their suitability to the project:
- **Decision Tree**
- **Random Forest**
- **LogisticRegression**
- **KNN**
- **SVC**

### 5. Baseline Model Training
- Trained baseline models using all features.
- Optimized hyperparameters for each algorithm.
- Selected the best baseline model based on its **ROC AUC score**.


## Setup Instructions

To set up this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Andrianarimanana/Aretim-po_Heart_Disease_Prediction

2. Activate virtual environment (make sure pipenv is already installed):
   ```bash
   pipenv shell

3. Install Dependencies:
   ```bash
   pipenv install

4. Activate the Virtual Environment
   ```bash
   pipenv shell

5. Run the project locally with pipenv
    ```bash
   # train the model
   pypenv python train.py

   # do prediction
   pipenv run python predict.py

To set up this projet using **Docker Container**

1. Build the docker image (make sure docker is already installed):
   ```bash
   docker build -t predict-app .

2. Running the docker container:
   ```bash
      docker run -it --rm -p 9696:9696 predict-app
   

