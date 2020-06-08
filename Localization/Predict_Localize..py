from keras.models import Model
from keras.layers import Input, Dense
from matplotlib import pyplot as plt
from PIL import Image
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from skimage import data_dir,io,transform,color
import pydot
from keras.utils import plot_model
import os
import h5py
from keras import regularizers
from keras.models import load_model
import matplotlib
import numpy as np
from PIL import Image, ImageDraw
# pretreatment 预处理
# input image dimensions
model = load_model("model.h5")


image_Test_path = "D:/MATLAB_Undergraduate Design/数据/裂缝识别用数据库/Test/"

Size = 1024
for i in range(1):
    Label_Matrix = np.zeros([64,64])
    RawImage = Image.open(image_Test_path + "Crack" + str(i+1) + ".jpg").convert('L')
    RawImage = RawImage.resize((Size, Size),Image.ANTIALIAS)
    draw = ImageDraw.Draw(RawImage)
    for j in range(64):
        for k in range(64):
            Crop_Image = RawImage.crop((16*j, 16*k, 16*(j+1), 16*(k+1)))
            Imag = np.expand_dims(np.array(Crop_Image), axis = 0)
            Imag = np.expand_dims(np.array(Imag), axis = -1)
            label = np.argmax(model.predict(Imag/255))
            if(label == 1):
                Label_Matrix[j,k] = 1
                draw.polygon([(16*j, 16*k), (16*(j+1), 16*k), (16*(j+1), 16*(k+1)), (16*j, 16*(k+1))], fill = None, outline=255)
    RawImage.save("C:/Users/10624/Desktop/CNN程序/裂缝定位结果/Crack%d.jpg"%(i+1))