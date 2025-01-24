import pickle
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_XGBClassifier.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('Heart Disease prediction service')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
            # Convert input data to a DataFrame
    X = pd.DataFrame([data])

        # Predict
    y_pred = model.predict(X)
    y_pred_class=int(y_pred[0])
    match = {0: 'no heart disease.', 1: 'Mild Heart Disease types.', 2: 'Moderate Heart Disease type.', 3: 'Heart Disease type.', 4: 'Critical Heart Disease type.'}
    y_pred_class = match[y_pred_class]
    return jsonify({'prediction': y_pred_class})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)