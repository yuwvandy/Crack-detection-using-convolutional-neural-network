from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from skimage import data_dir,io,transform,color
import numpy as np

batch_size = 10
num_classes = 2
epochs = 30

# pretreatment 预处理
# input image dimensions
img_rows, img_cols = 100, 100

# the data, shuffled and split between train and test sets
image_Train_path = "D:\MATLAB_Undergraduate Design\Train"
image_Test_path = "D:\MATLAB_Undergraduate Design\Test"
def convert_gray(f):
 rgb=io.imread(f)    #依次读取rgb图片
 gray=color.rgb2gray(rgb)   #将rgb图片转换成灰度图
 dst=transform.resize(gray,(100,100))  #将灰度图片大小转换为256*256
 return dst
	
	
str_Train=image_Train_path+'/*.jpg'
str_Test=image_Test_path+'/*.jpg'

coll_Train = io.ImageCollection(str_Train,load_func=convert_gray)
coll_Test = io.ImageCollection(str_Test,load_func=convert_gray)

x_train = np.array(coll_Train)
y_train = np.hstack((np.ones(363),np.zeros(263)))

x_test = np.array(coll_Test)
y_test = np.hstack((np.ones(24),np.zeros(24)))

if K.image_data_format() == 'channels_first': #判断图片格式是 channel在前还是在后（channel：黑白为1,彩色为3）
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) #shape[0]指例子的个数
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# build the neural net 建模型(卷积—relu-卷积-relu-池化-relu-卷积-relu-池化-全连接)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',strides=(1,1),padding='same',input_shape=input_shape)) # 32个过滤器，过滤器大小是3×3，32×26×26
model.add(Conv2D(64,kernel_size=(3, 3),activation='relu',strides=(1,1),padding='same')) # 32个过滤器，过滤器大小是3×3，32×26×26
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='valid'))# 向下取样
model.add(Conv2D(64,kernel_size=(3, 3),activation='relu',strides=(1,1),padding='same')) # 32个过滤器，过滤器大小是3×3，32×26×26
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='valid'))# 向下取样
model.add(Dropout(0.25))
model.add(Flatten()) #降维：将64×12×12降为1维（即把他们相乘起来）
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) #全连接2层


# compile the model 编译模型
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# train the model 训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# test the model 测试模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])