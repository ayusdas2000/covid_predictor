import numpy as np
import os
from keras.models import *
from keras.preprocessing import image
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras import backend as K



# Create your views here.
def index(request):
    K.clear_session()
    result = 'negative'
    if request.method =='POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        fs.save(name=uploaded_file.name,content=uploaded_file,max_length=None)
        predict = load_model('model_adv.h5',compile=False)
        #img = image.load_img(uploaded_file, target_size=(224, 224))
        path = os.path.join('./media',uploaded_file.name)
        img = image.load_img(path,target_size=(224, 224))
        print("checking file type: ", type(img))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        p = predict.predict_classes(img)
        os.remove(path)
        print('this is working: ',type(p[0][0]))
        print('also printing reuslt ',p[0][0])
        if p[0][0]==0:
            result='positive'
        K.clear_session()
    return render(request, 'corona/index.html',{'result':result})
