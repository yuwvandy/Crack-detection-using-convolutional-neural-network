from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from skimage import data_dir,io,transform,color
import numpy as np
import pydot
from keras.utils import plot_model
import os
import h5py
from matplotlib import pyplot as plt
from keras import regularizers
from keras.preprocessing import image
from keras.models import load_model
from scipy import misc 
from keras import models

layeroutput = 10
x_ex = np.expand_dims(x_train[963], axis = 0)
channel = [32, 32, 32, 32, 64, 64, 64]

def Layer_Output(img, model, layeroutput, p, channel):
    layer_outputs = [layer.output for layer in model.layers[:layeroutput]]
    activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
    activations = activation_model.predict(img)
    first_layer_activation = activations[p]
    plt.matshow(first_layer_activation[0, :, :, channel], cmap='viridis')
    plt.savefig('C:/Users/10624/Desktop/CNN程序/Experiment/Ex1/卷积核输出/Crack964/'+str(p+1)+ 'Layer' + str(i+1)+'.png')
    
 

for p in range(7):
    for i in range(channel[p]):
        Layer_Output(x_ex, model, layeroutput, p, i)