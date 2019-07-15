from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import numpy as np
import pandas as pd
import os
import json
import time
from  pages.new_SAX import get_segment_data


def csv_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("data.csv", myfile)
        uploaded_file_url = fs.url(filename)
        time.sleep(2)
        return redirect('brush_graph')
    return render(request, 'csv_upload.html')



def brush_graph(request):
    csv_data= read_csv()
    data={
    "key" : "csv",
    "value": csv_data
    }


    if request.method == 'POST' :
        start = request.POST.get('start')
        end = request.POST.get('end')
        get_segment_data(csv_data,start,end)
        return redirect('images')

    return render(request, 'brush_graph.html',  data )




def home(request):
    return render(request, 'test.html',  {} )




def images_display(request):

    path="C:/Megatron/Thesis/Thesis_Work/Django/Thesis/Motif_Discovery/static/ops"
    img_list =os.listdir(path)

    #os.remove("static/ops/.png")
    #print("File Removed!")

    return render(request, 'images.html', {'images': img_list})


def read_csv():
        data =  pd.read_csv('files/data.csv', header=None,usecols=[1],skiprows=1)
        x1 = np.asfarray(data.values.flatten(),float)
        x1= x1.tolist()

        #os.remove("files/data.csv")
        #print("File Removed!")

        return x1
