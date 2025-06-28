import numpy as np
from flask import Flask, request, render_template
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# path = "C:\\Users\\psiri\\Desktop\\OptiCrop\\"
# model = pickle.load(open(path + "model.pkl", 'rb'))

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
    app.run(host='0.0.0.0', port=5000, debug=True)