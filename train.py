#!/usr/bin/env python
# coding: utf-8

# #### Project Aretim-po Prediction or Heart Disease Prediction
# The porpuse of this project is a machine learning focused on forcasting thunderstorms in northern Madagascar, particularly around Nosy Be. The project aims to provide accurate short-term predictions (0–6 hours) to mitigate risks, protect lives, and support emergency responses in this vulnerable region.

# #### Data importation
import sys

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt

from IPython import get_ipython

# Initialiser IPython si nécessaire
ipython = get_ipython()
if ipython is None:
    from IPython.terminal.embed import InteractiveShellEmbed
    ipython = InteractiveShellEmbed()

train_df = pd.read_csv('./Data/heart_disease_uci.csv')


# #### Data exploration

train_df.head(3)
#train_df.describe()
print("Data exploration")

# #### Data Preparation and Features importance

print("------------- Apply feature engineering --------")
# Apply feature engineering


# Define lag features and intervals


train_df.columns

#train_df.describe()

print("------------- Prepare Training data --------")
# #### Prepare Training data

# Prepare Training data
feature_cols = [
    'hour_sin', 'hour_cos', 
    'distance_x', 'distance_y', 'intensity', 'size', 'distance',
    'is_peak_cyclone_season', 
    'cyclone_season_weight', 'peak_cyclone_daytime_interaction', 'size_change_30', 'bearing'
]

# Add lag columns to feature_cols

print(" --- Split training and validation sets ----")
# #### Split training and validation sets


# x_full_train, x_test,y1h_full_train, y1h_test= train_test_split(X, y_1h, test_size=0.2, random_state=11)
# x_train, x_val, y1h_train, y1h_val= train_test_split(x_full_train,y1h_full_train, test_size=0.25, random_state=11)

# _, _, y3h_full_train, y3h_test = train_test_split(X, y_3h, test_size=0.2, random_state=11)
#_, _, y3h_train, y3h_val= train_test_split(x_full_train,y3h_full_train, test_size=0.25, random_state=11)

# #### trainning the Modele
df_train = x_train.reset_index(drop=True)
df_val = x_val.reset_index(drop=True)
df_test = x_test.reset_index(drop=True)
df_full_train = x_full_train.reset_index(drop=True)

print("trainning the Modele -- model making")
# ##### Decision Tree

train_dicts = df_train.fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

dt_model_1h = DecisionTreeClassifier()
dt_model_1h.fit(X_train, y1h_train)

val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = dt_model_1h.predict_proba(X_val)[:, 1]
roc_auc_score(y1h_val, y_pred)

y_pred = dt_model_1h.predict_proba(X_train)[:, 1]
roc_auc_score(y1h_train, y_pred)

dt_model_1h = DecisionTreeClassifier(max_depth=2)
dt_model_1h.fit(X_train, y1h_train)

y_pred = dt_model_1h.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y1h_train, y_pred)
print('train:', auc)

y_pred = dt_model_1h.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y1h_val, y_pred)
print('val:', auc)

# ##### Decision tree Tunning
# selecting max_depth
# selecting min_samples_leaf

print(" Decision tree Tunning")
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, None]

for depth in depths: 
    dt_model_1h = DecisionTreeClassifier(max_depth=depth)
    dt_model_1h.fit(X_train, y1h_train)
    
    y_pred = dt_model_1h.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y1h_val, y_pred)
    
    print('%4s -> %.3f' % (depth, auc))


scores = []

for depth in [5, 6, 7]:
    for s in [1, 5, 10, 15, 20, 500, 100, 200]:
        dt_model_1h = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
        dt_model_1h.fit(X_train, y1h_train)

        y_pred = dt_model_1h.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y1h_val, y_pred)
        
        scores.append((depth, s, auc))

columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)

sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")
plt.show()

dt_model_1h = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)
dt_model_1h.fit(X_train, y1h_train)


print(export_text(dt_model_1h, feature_names=list(dv.get_feature_names_out())))


# ##### Trainning random forest
print("Trainning random forest")



# ##### Save the model
print("Save the model")

# Saving the model with pickle
with open('model_xboost.bin', 'wb') as file:
    pickle.dump((dv,model), file)

print("model at : model_xboost.bin ")
