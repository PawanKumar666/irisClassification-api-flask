import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('irisModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if(prediction[0] == 0):
        output = "Iris_Setosa"
    elif(prediction[0] == 1):
        output = "Iris_Versicolor"
    else:
        output = "Iris_Virginica"

    return render_template('index.html', prediction_text='The breed corresponding to given characteristics is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
