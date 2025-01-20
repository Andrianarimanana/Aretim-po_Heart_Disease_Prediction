# Aretim-po Prediction (Heart Disease) 

Welcome to the **Aretim-po Prediction (Heart Disease) ** project! This repository designed a project to predict heart disease risk comparing 3 Machine Learning models, featuring data preprocessing, model training, evaluation, and deployment-ready insights.
![Heart Disease Prediction](Image/Young-Myocarditis-Heart-Concept.webp)
## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model performance](#model)
3. [Deploiment](#deploiment)
3. [Setup Instructions](#setup-instructions)

## Overview

The porpuse of this project is a machine learning focused on forcasting 

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

## Purpose
The dataset is primarily used for:

1. **Prediction**: Determine if a patient has heart disease based on their attributes.
2. **Exploration**: Analyze the dataset to uncover insights that improve understanding of heart disease.

## Notes
- The full database contains 76 attributes, but only these 14 are  used .

Files in Repository:
heart_disease_uci.csv: Contains input features and labels for model training.
1. notebook.ipynb :
2. train.py :
3. predict.py :
4. predict-test.py :
5. Pipfile.lock :
6. Pipfile :
7. Dockerfile :

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

To set up this projet using Docker Container

1. Build the docker image (make sure docker is already installed):
   ```bash
   docker build -t predict-app .

2. Running the docker container:
   ```bash
      docker run -it --rm -p 9696:9696 predict-app
   

