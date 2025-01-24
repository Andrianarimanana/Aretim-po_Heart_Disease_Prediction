#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

User_id = 'xyz-123'

feature = {
    "thal":2,
    "slope":1,
    "fbs":0,
    "exang":1,
    "restecg":1,
    "id":224,
    "age":53,
    "sex":1,
    "dataset":0,
    "cp":0,
    "trestbps":123.0,
    "chol":282.0,
    "thalch":95.0,
    "oldpeak":2.0,
    "ca":2.0
    }
response = requests.post(url, json=feature).json()
print(response)

# if response['churn'] == True:
#     print('sending promo email to %s' % meteo_id)
# else:
#     print('not sending promo email to %s' % meteo_id)