from django.shortcuts import render

# Create your views here.
# -*- coding: utf-8 -*-
import pickle
from .preprocessing import text_Preprocessing
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# load the model from disk
clf = pickle.load(open('sentiment/models/nb_clf.pkl', 'rb'))
cv=pickle.load(open('sentiment/models/tfidf_model.pkl','rb'))




def home(request):
	return render(request, 'home.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        if(len(message)>2):
            text = [message]
            data = text_Preprocessing(text)
            vect = cv.transform(data)
            my_prediction = clf.predict(vect)
        else:
            my_prediction=3
        return render(request,'home.html',{'prediction':my_prediction})
    else:
        return render(request,'home.html',{})




