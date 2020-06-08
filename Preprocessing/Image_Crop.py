import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

Size = 512

image_Train_Crack_path = "D:/MATLAB_Undergraduate Design/数据/新数据/Train/Crack/"
image_Train_UnCrack_path = "D:/MATLAB_Undergraduate Design/数据/新数据/Train/UnCrack/"
image_Test_path = "D:/MATLAB_Undergraduate Design/Test/"
p = 1

RawImage = Image.open(image_Train_Crack_path + "Crack" + str(312) + ".jpg")
RawImage = RawImage.resize((Size, Size),Image.ANTIALIAS)
for j in range(32):
    for k in range(32):
        Crop_Image = RawImage.crop((16*j, 16*k, 16*(j+1), 16*(k+1)))
        Crop_Image.save("D:/MATLAB_Undergraduate Design/Train_dataaug/Crack312312" + str(p) + ".jpg")
        p+=1

p = 1
for i in range(626-363):
    RawImage = Image.open(image_Train_UnCrack_path + "UnCrack" + str(i+1) + ".jpg")
    RawImage = RawImage.resize((Size, Size),Image.ANTIALIAS)
    for j in range(4):
        for k in range(4):
            Crop_Image = RawImage.crop((256*j, 256*k, 256*(j+1), 256*(k+1)))
            Crop_Image.save("D:/MATLAB_Undergraduate Design/Train_dataaug/UnCrack" + str(p) + ".jpg")
            p+=1