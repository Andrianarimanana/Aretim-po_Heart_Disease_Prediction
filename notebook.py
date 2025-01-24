#!/usr/bin/env python
# coding: utf-8

# #### Project Aretim-po (Heart Disease Prediction)
# This repository designed a project to predict heart disease risk comparing 3 ML models, featuring data preprocessing, model training, evaluation, and deployment-ready insights.

# #### Data importation

import pandas as pd
import numpy as np
import os
import sys
import pickle

import seaborn as sns
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import ListedColormap

# 3. To preprocess the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# 4. import Iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 5. Machine Learning
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score

# 6. For Classification task.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier
# from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier

# 7. Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 8. Ignore warnings
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('./Data/heart_disease_uci.csv')
#status='DEV'
status='PROD'

train_df.head(3)

train_df.describe()



train_df.info()
print("Data Preparation and Features importance_______________________________")

# #### Data Preparation and Features importance
# 

train_df.shape

train_df.columns

train_df['age'].describe()


custom_colors = ["#226D68", "#ECF8F6", "#FEEAA1"]  
sns.histplot(train_df['age'], kde=True, color="#18534F", palette=custom_colors)

print("The distribution of the values in the 'Age' column appears to follow a normal distribution.")

# The distribution of the values in the "Age" column appears to follow a normal distribution.

train_df['sex'].value_counts()


# Find the values count of age column grouping by sex column
print("# Find the values count of age column grouping by sex column")
train_df.groupby('sex')['age'].value_counts()


train_df['dataset'].value_counts()


print('plot the countplot of dataset column')

try:
    # Vérifier si le script est exécuté dans un environnement Jupyter
    if "ipykernel" in sys.modules:
        # Si Visual Studio Code est utilisé avec Jupyter
        if os.getenv("VSCODE_PID"):
            print( "Visual Studio Code")
            import plotly.io as pio
            pio.renderers.default = 'browser'
        else:
            print( "Jupyter Notebook")
    else:
        print( "Script non exécuté dans un notebook")
except Exception as e:
    print(f"Erreur lors de la détection : {e}")


if status=="DEV":
    import plotly.io as pio
    pio.renderers.default = 'browser'
    #pio.renderers.default = 'notebook'

fig =px.bar(train_df, x='dataset', color='sex')

fig.show()

# print the values of dataset column groupes by sex
#print (train_df.groupby('sex')['dataset'].value_counts())


#  plot the histogram of age column using plotly and coloring this by sex

fig = px.histogram(data_frame=train_df, x='age', color= 'sex')
fig.show()


# make a plot of age column using plotly and coloring by dataset

fig = px.histogram(data_frame=train_df, x='age', color= 'dataset')
fig.show()

# print the mean median and mode of age column grouped by dataset column
print("___________________________________________________________")
print ("Mean of the dataset: ",train_df.groupby('dataset')['age'].mean())
print("___________________________________________________________")
print ("Median of the dataset: ",train_df.groupby('dataset')['age'].median())
print("___________________________________________________________")
print ("Mode of the dataset: ",train_df.groupby('dataset')['age'].agg(pd.Series.mode))
print("___________________________________________________________")


# Exploring CP (Chest Pain) column

# value count of cp column
train_df['cp'].value_counts()

sns.countplot(train_df, x='cp', hue= 'sex')


# 

# count plot of cp column by dataset column
sns.countplot(train_df,x='cp',hue='dataset')


# Draw the plot of age column group by cp column

fig = px.histogram(data_frame=train_df, x='age', color='cp')
fig.show()


# Let's explore the trestbps (resting blood pressure) column:
# The normal resting blood pressure is 120/80 mm Hg

# lets summerize the trestbps column
train_df['trestbps'].describe()


# ##### Handling missing values in trestbps column
print("Handling missing values in trestbps column")
# There are some missing values becuase total values is 920 but here we have 861

# Dealing with Missing values in trestbps column.
# find the percentage of misssing values in trestbps column
print(f"Percentage of missing values in trestbps column: {train_df['trestbps'].isnull().sum() /len(train_df) *100:.2f}%")


# Removing missing values using Iterative imputer
print("Removing missing values using Iterative imputer")
# Impute the missing values of trestbps column using iterative imputer
# create an object of iteratvie imputer
imputer1 = IterativeImputer(max_iter=10, random_state=42)

# Fit the imputer on trestbps column
imputer1.fit(train_df[['trestbps']])

# Transform the data
train_df['trestbps'] = imputer1.transform(train_df[['trestbps']])

# Check the missing values in trestbps column
print(f"Missing values in trestbps column: {train_df['trestbps'].isnull().sum()}")


# let's see which columns has missing values
(train_df.isnull().sum()/ len(train_df)* 100).sort_values(ascending=False)


# create an object of iterative imputer 
imputer2 = IterativeImputer(max_iter=10, random_state=42)

# fit transform on ca,oldpeak, thal,chol and thalch columns
train_df['ca'] = imputer2.fit_transform(train_df[['ca']])
train_df['oldpeak']= imputer2.fit_transform(train_df[['oldpeak']])
train_df['chol'] = imputer2.fit_transform(train_df[['chol']])
train_df['thalch'] = imputer2.fit_transform(train_df[['thalch']])
# let's check again for missing values
(train_df.isnull().sum()/ len(train_df)* 100).sort_values(ascending=False)


# #### All the coloumns are imputed which has floating data types and now lets impute the columns which has object data type.

print(f"The missing values in thal column are: {train_df['thal'].isnull().sum()}")


train_df['thal'].value_counts()


train_df.tail()

# find missing values.
train_df.isnull().sum()[train_df.isnull().sum()>0].sort_values(ascending=False)


missing_data_cols = train_df.isnull().sum()[train_df.isnull().sum()>0].index.tolist()

missing_data_cols

# find categorical Columns
cat_cols = train_df.select_dtypes(include='object').columns.tolist()
cat_cols

# find Numerical Columns
Num_cols = train_df.select_dtypes(exclude='object').columns.tolist()
Num_cols

print(f'categorical Columns: {cat_cols}')
print(f'numerical Columns: {Num_cols}')


# FInd columns 
categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']
bool_cols = ['fbs', 'exang']
numerical_cols = ['oldpeak', 'thalch', 'chol', 'trestbps', 'age']


# ##### Dealing missing Values with Machine learning model
print("Dealing missing Values with Machine learning model")
passed_col = categorical_cols
def impute_categorical_missing_data(passed_col):
    
    df_null = train_df[train_df[passed_col].isnull()]
    df_not_null = train_df[train_df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    
    label_encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    if passed_col in bool_cols:
        y = label_encoder.fit_transform(y)
        
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
    
    X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(x_val)

    acc_score = accuracy_score(y_val, y_pred)

    print("The feature '"+ passed_col+ "' has been imputed with", round((acc_score * 100), 2), "accuracy\n")

    X = df_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
                
    if len(df_null) > 0: 
        df_null[passed_col] = rf_classifier.predict(X)
        if passed_col in bool_cols:
            df_null[passed_col] = df_null[passed_col].map({0: False, 1: True})
        else:
            pass
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])
    
    return df_combined[passed_col]

def impute_continuous_missing_data(passed_col):
    
    df_null = train_df[train_df[passed_col].isnull()]
    df_not_null = train_df[train_df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    
    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
    
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
    
    X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor()

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(x_val)

    print("MAE =", mean_absolute_error(y_val, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_val, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_val, y_pred), "\n")

    X = df_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
                
    if len(df_null) > 0: 
        df_null[passed_col] = rf_regressor.predict(X)
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])
    
    return df_combined[passed_col]


train_df.isnull().sum().sort_values(ascending=False)

# impute missing values using our functions
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((train_df[col].isnull().sum() / len(train_df)) * 100, 2))+"%")
    if col in categorical_cols:
        train_df[col] = impute_categorical_missing_data(col)
    elif col in numeric_cols:
        train_df[col] = impute_continuous_missing_data(col)
    else:
        pass


train_df.isnull().sum().sort_values(ascending=False)


# Now, all columns are complete without any missing data.
print("Now, all columns are complete without any missing data.")
# ##### Dealing with outliers
print("Dealing with outliers")

print("_________________________________________________________________________________________________________________________________________________")

sns.set(rc={"axes.facecolor":"#87CEEB","figure.facecolor":"#EEE8AA"})  # Change figure background color

palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

plt.figure(figsize=(10,8))

for i, col in enumerate(numerical_cols):
    plt.subplot(3,2, i+1)
    sns.boxenplot(x=train_df[col], color=palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(col)
    
plt.show()


# print the row from train_df where trestbps value is 0
train_df[train_df['trestbps']==0]


# Remove the column because it is an outlier because trestbps cannot be zero.
train_df=train_df[train_df['trestbps']!=0]


# confirm


sns.set(rc={"axes.facecolor":"#B76E79","figure.facecolor":"#C0C0C0"})
modified_palette = ["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"]
cmap = ListedColormap(modified_palette)

plt.figure(figsize=(10,8))



for i, col in enumerate(numerical_cols):
    plt.subplot(3,2, i+1)
    sns.boxenplot(x=train_df[col], color=modified_palette[i % len(modified_palette)])  # Use modulo to cycle through colors
    plt.title(col)
    
plt.show()



train_df.trestbps.describe()


# ##### Handling Oldpeak Outliers

# Set facecolors
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})

# Define the "night vision" color palette
night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]

# Use the "night vision" palette for the plots
plt.figure(figsize=(10, 8))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 2, i + 1)
    sns.boxenplot(x=train_df[col], color=night_vision_palette[i % len(night_vision_palette)])  # Use modulo to cycle through colors
    plt.title(col)

plt.show()


# let's remove -2 

train_df=train_df[train_df['oldpeak'] >=-1]


# Set facecolors
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})

# Define the "night vision" color palette
night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]

# Use the "night vision" palette for the plots
plt.figure(figsize=(10, 8))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 2, i + 1)
    sns.boxenplot(x=train_df[col], color=night_vision_palette[i % len(night_vision_palette)])  # Use modulo to cycle through colors
    plt.title(col)

plt.show()


# ##### Handling Outliers in Age Column
# Minimum age is 31 to have chest pain which can be possible so its not an outlier.
# max age is 77 which is also possible so its not an outlier as well.

train_df.age.describe()


# ##### Handling trestbps column outliers
print("Handling trestbps column outliers")

palette = ["#999999", "#666666", "#333333"]

sns.histplot(data=train_df, 
             x='trestbps', 
             kde=True,
             color=palette[0])

plt.title('Resting Blood Pressure')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = palette[1]
plt.rcParams['axes.facecolor'] = palette[2] 
plt.show()


# create a histplot trestbops column to analyse with sex column
sns.histplot(train_df, x='trestbps', kde=True, palette = "Spectral", hue ='sex') 
plt.show()


# Every things seems OK 

# #### Prepare Training data
print("Prepare Training data")
# Prepare Training data
feature_cols = ['thal',
'slope',
'fbs',
'exang',
'restecg',
'id',
'age',
'sex',
'dataset',
'cp',
'trestbps',
'chol',
'thalch',
'oldpeak',
'ca'
]
X = train_df[feature_cols]
y = train_df['num']

# Encode the categorical columns

Label_Encoder = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
        X[col] = Label_Encoder.fit_transform(X[col])
    else:
        pass


# #### Split training and validation sets
print("Split training and validation sets")
x_full_train, x_test,y_full_train, y_test= train_test_split(X, y, test_size=0.2, random_state=11)
x_train, x_val, y_train, y_val= train_test_split(x_full_train,y_full_train, test_size=0.25, random_state=11)

# #### Train the model

print(x_full_train.size)
print(x_train.size)
print(x_test.size)
print(x_val.size)

# ##### Find the Best Models & hyperparameter tuning using GridSearchCV 
print("Find the Best Models & hyperparameter tuning using GridSearchCV")
# Models to optimize
models = {
    'KNN': KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(),
    'XGBoost': XGBClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVC': SVC()
}

param_grids = {
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
     'DecisionTree': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    },
        'XGBoost': {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7]
    },
    'GradientBoosting': {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7]
    },
    'SVC': {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    }
}


best_models = {}
best_scores = {}
# x_train, x_val, y_train, y_val
for model_name, model in models.items():
    print(f"Optimizing {model_name}...")
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    best_models[model_name] = grid_search.best_estimator_
    best_scores[model_name] = grid_search.best_score_

    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best Score for {model_name}: {grid_search.best_score_}")

    y_pred = grid_search.best_estimator_.predict(x_val)
    test_accuracy = accuracy_score(y_val, y_pred)
    print(f"Test Accuracy for {model_name}: {test_accuracy}\n")

    # Classification Report
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_val, y_pred))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{conf_matrix}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    class_labels = sorted(set(y_val)) 
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# This evaluation concludes that XGBoost is the most suitable model for the current dataset, given its strong performance across key metrics and potential for further optimization.
print("This evaluation concludes that XGBoost is the most suitable model for the current dataset, given its strong performance across key metrics and potential for further optimization.")
# ##### Selecting the final model
print("Selecting the final model ")

models = {
    'XGBoost': XGBClassifier()
}

param_grids = {
    'XGBoost': {
        'learning_rate': 0.075,
        'max_depth': 5,
        'n_estimators':50
    }

}


models = {
    'XGBoost': XGBClassifier()
}

param_grids = {
    'XGBoost': {
        'learning_rate': [0.075, 0.1, 0.15],
        'max_depth': [4, 5, 6],
        'n_estimators': [50, 75, 100]
    }

}


best_models = {}
best_scores = {}

x_train=x_full_train
x_val=x_test
y_train=y_full_train
y_val=y_test
for model_name, model in models.items():
    print(f"Optimizing {model_name}...")
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    best_models[model_name] = grid_search.best_estimator_
    best_scores[model_name] = grid_search.best_score_

    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best Score for {model_name}: {grid_search.best_score_}")

    y_pred = grid_search.best_estimator_.predict(x_val)
    test_accuracy = accuracy_score(y_val, y_pred)
    print(f"Test Accuracy for {model_name}: {test_accuracy}\n")

    # Classification Report
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_val, y_pred))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{conf_matrix}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    class_labels = sorted(set(y_val)) 
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# We will Train  the project with XGBOOST Modele that win
print("We will Train  the project with XGBOOST Modele that win")

x_train=x_full_train
y_train=y_full_train
x_val=x_test

y_val=y_test
param_grids = {
    'learning_rate': 0.075,
    'max_depth': 4,
    'n_estimators': 50
}

model = XGBClassifier(**param_grids)

# Entraînement du modèle
model.fit(x_train, y_train)

# Évaluation du modèle
y_pred = model.predict(x_val)
# test_accuracy = accuracy_score(y_val, y_pred)
# print(f"Accuracy: {test_accuracy:.2f}")
model_name="XGBClassifier"
test_accuracy = accuracy_score(y_val, y_pred)
print(f"Test Accuracy for {model_name}: {test_accuracy}\n")
# Classification Report
print(f"Classification Report for {model_name}:\n")
print(classification_report(y_val, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)
print(f"Confusion Matrix for {model_name}:\n{conf_matrix}")
# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
class_labels = sorted(set(y_val)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Confusion Matrix for {model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# ##### Save the model
print("Save the model") 



# Saving the model with pickle
with open('model_XGBClassifier.bin', 'wb') as file:
    pickle.dump((model), file)


print("Modele saved in model_XGBClassifier.bin .... done")