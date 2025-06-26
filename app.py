import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'model.pkl'), 'rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    output = prediction [0]
    return render_template('findyourcrop.html',prediction_text='Best crop for given conditions is {}'.format(output))

if __name__ == "__main__":
    app.run(debug = True, use_reloader = False)