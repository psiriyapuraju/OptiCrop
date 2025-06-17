#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from flask import Flask, request, render_template
import pickle


# In[11]:


app = Flask(__name__)

path = r'C:\Users\psiri\Desktop\OptiCrop\\'
model = pickle.load(open(path + 'model.pkl','rb'))


# In[13]:


@app.route('/')

def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html')


# In[15]:


@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    output = prediction [0]
    return render_template('findyourcrop.html',prediction_text='Best crop for given conditions is {}'.format(output))


# In[ ]:


if __name__ == "__main__":
    app.run(debug = True, use_reloader = False)


# In[ ]:




