from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
from CountryNumerical import Numerical_country

model = pickle.load(open('migration.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('deploy.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.form['Gender']
    if data == 'Female':
        data1 = 0
    else:
        data1 = 1
    data2 = request.form['Last2']
    data3 = request.form['Last1']
    data4 = request.form['Country']
    data4_correct = Numerical_country(data4)
    inputs = np.array([data1,data2,data3,data4_correct]).reshape(1,-1)
    pred = int(model.predict(inputs)[0])
    return render_template('deploy.html',prediction_text = 'No.of people migrated to {} from India are: {}'.format(data4,pred))

if __name__ == '__main__':
    app.run(debug=True)
