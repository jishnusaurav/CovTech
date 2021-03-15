from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import matplotlib.pyplot as plt
from fastai.tabular.all import *
import pandas as pd
import matplotlib.pyplot as plt


def covid1(request):
    df = pd.read_csv('static/covid11.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:]
    value = ''

    if request.method == 'POST':        
        age = float(request.POST['temp'])
        sex = float(request.POST['bp'])
        cp = float(request.POST['age'])
        trestbps = float(request.POST['nose'])
        chol = float(request.POST['breath'])
        chol1 = float(request.POST['ht'])
        chol2= float(request.POST['a'])
        chol3 = float(request.POST['b'])
        chol4 = float(request.POST['c'])
        chol5 = float(request.POST['d'])
        chol6 = float(request.POST['e'])
        chol7 = float(request.POST['f'])

        data = np.array(
            (age,
             sex,
             cp,
             trestbps,
             chol,chol1,chol2,chol3,chol4,chol5,chol6,2,2,2
            )
        ).reshape(1, 14)
        rand_forest=joblib.load("/home/jishnusaurav/Downloads/priovax-main/machine-learning/model.pkl")
        predictions = rand_forest.predict(data)
        x=str(predictions[0])
        print(predictions[0])
        print("123")
        if(x=="1"):
            x="You should get vaccinated!"
        else:
            x="You need not get vaccinated right now!"
        return render(request,
                  'diseasepredictor/rforest.html',
                  {
                      'context': x
                  })
    else:
        return render(request,
                  'diseasepredictor/rforest.html',
                  {
                      'context': "No data"
                  })
def covid2(request):
    df = pd.read_csv('static/covid22.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    print(X.shape, Y.shape)

    value = ''
    if request.method == 'POST':

        temp = request.POST['temp']
        age = request.POST['age']
        bp = request.POST['bp']
        nose = request.POST['nose']
        breath = request.POST['breath']

        if bp=='Yes' or bp == 'yes' or bp=='YES':
            bp=1
        elif bp=='No' or bp == 'no' or bp=='NO':
            bp=0
        
        else:
            return render(request,
                  'diseasepredictor/rforest.html',
                  {
                      'context': value,
                      'error':"Please enter correct data"
                })
        
        if nose=='Yes' or nose == 'yes' or nose=='YES':
            nose=1
        elif nose=='No' or nose == 'no' or nose=='NO':
            nose=0
        else:
            return render(request,
                  'diseasepredictor/knn.html',
                  {
                      'context': value,
                      'error':"Please enter correct data"
                })
        if breath=='Yes' or breath == 'yes' or breath=='YES':
            breath=1
        elif breath=='No' or breath == 'no' or breath=='NO':
            breath=0
        else:
            return render(request,
                  'diseasepredictor/knn.html',
                  {
                      'context': value,
                      'error':"Please enter correct data"
                })

        user_data = np.array(
            (temp,
             bp,
             age,
             nose,
             breath
            )
        ).reshape(1, 5)


        rf = KNeighborsClassifier()
        rf.fit(np.nan_to_num(X), Y)

        predictions = rf.predict(user_data)
        print(predictions)

        if int(predictions[0]) == 1:
            value = 'You May have COVID-19 Virus. Kindly get in contact with a Doctor!!!'
        elif int(predictions[0]) == 0:
            value = "You are SAFE!!!"

    return render(request,
                  'diseasepredictor/knn.html',
                  {
                      'context': value
                  })


def home(request):

    return render(request,
                  'diseasepredictor/predict.html')



# def handler404(request):
#     return render(request, '404.html', status=404)
