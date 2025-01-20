#### Aretim-po_Heart_Disease_Prediction
This repository designed a project to predict heart disease risk comparing 3 ML models, featuring data preprocessing, model training, evaluation, and deployment-ready insights.
# Aretim-po Prediction (Heart Disease) 

Welcome to the **Aretim-po Prediction (Heart Disease) ** project! This repository provides tools and resources for predicting 
![Heart Disease Prediction](Image/Young-Myocarditis-Heart-Concept.webp)
## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Setup Instructions](#setup-instructions)

## Overview

The porpuse of this project is a machine learning focused on forcasting 

## Dataset

The repository includes data derived from [EUMETSAT](https://www.eumetsat.int/) [Klein et al. (2018)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017JD027432),

Dataset Details:

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
   

