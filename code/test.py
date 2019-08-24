#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:33:42 2019

@author: qq
"""

#test
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import keras
from keras.datasets import mnist
import random
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import matplotlib.image as mp
from PIL import Image
from skimage import io

#x_test是无数张图片，i是第i张
def add_9_white_dot(x_test,i,x_axis,y_axis):
    x_test[i][int(x_axis) - 1][int(y_axis) - 1] = 255
    x_test[i][int(x_axis) - 1][int(y_axis) ] = 255
    x_test[i][int(x_axis) - 1][int(y_axis) + 1] = 255
    x_test[i][int(x_axis) ][int(y_axis) - 1] = 255
    x_test[i][int(x_axis) ][int(y_axis) ] = 255
    x_test[i][int(x_axis) ][int(y_axis) + 1] = 255
    x_test[i][int(x_axis) +1][int(y_axis) - 1] = 255
    x_test[i][int(x_axis) +1][int(y_axis) ] = 255
    x_test[i][int(x_axis) +1][int(y_axis) +1] = 255
    return x_test


def subsample(x_test,i):
    for j in range(x_test[i].shape[0]):
        if j%2==0:#j是偶数
            for k in range(x_test[i].shape[1]):
                    x_test[i][j][k]=x_test[i][j+1][k]
    return x_test


def find_pass(model,image,test_label):
    pred=model.predict(image)
    pred=list(map(lambda x:np.argmax(x),pred))
    passlist=[]
    for i in range(len(image)):
        if pred[i]==test_label[i]:
            passlist.append(i)
    #print 'notpass:',len(notpasslist)
    return passlist

def saveimage(images,affected_list,y_test):
    for i in affected_list:
        imagename="../mutated/"+"mutant"+str(i)+"_"+str(y_test[i])+".jpg"
        plt.imsave(imagename,images[i].reshape(28,28),vmin=0, vmax=255,format="jpg",cmap='gray')
    return

def saveimage_origin(images,affected_list,y_test):
    for i in affected_list:
        imagename="../origin/"+"origin"+str(i)+"_"+str(y_test[i])+".jpg"
        plt.imsave(imagename,images[i].reshape(28,28),vmin=0, vmax=255,format="jpg",cmap='gray')
    return

#一个是变暗，估计影响不大
#旋转，估计影响大，但是需要打印出来看
#在白色的地方加一些黑色断点。

def darker(images):
    tmp=images
    for k in range(len(images)):
        for i in range(images[0].shape[0]):
            for j in range(images[0].shape[1]):
                tmp[k][i][j]=images[k][i][j]/10

    return tmp

def rotate(images):
    #先写往右转
    #以14*14为中心点，
    tmp=images.copy()

    for k in range(len(images)):
        for distance in range(13):
            #images[0][18]=[i for i in range(28)]    
            tmp[k] = rotate_with_center(tmp[k],14,14,distance+1)
            #tmp[k] 肯定更新了，不用怀疑了
    return tmp  
        
    

def rotate_with_center(image,center_x,center_y,distance):
    tmp=image.copy()
    #上
    for i in range(distance+1):
        tmp[center_x-distance][center_y-1+i] = image[center_x-distance][center_y+i]#上右
    for i in range(distance-1): 
        tmp[center_x-distance][center_y-i-2] = image[center_x-distance][center_y-i-1]#上左
        
    #下
    for i in range(distance):
        tmp[center_x+distance][center_y+1+i] = image[center_x+distance][center_y+i]#下右
        tmp[center_x+distance][center_y-i] = image[center_x+distance][center_y-i-1]#下左
    #左
    for i in range(distance+1): 
        tmp[center_x-i+1][center_y-distance] = image[center_x-i][center_y-distance]#左上
    for i in range(distance-1): 
        tmp[center_x+i+2][center_y-distance] = image[center_x+i+1][center_y-distance]#左下
    #右
    for i in range(distance): 
        tmp[center_x-i-1][center_y+distance] = image[center_x-i][center_y+distance]#右上
        tmp[center_x+i][center_y+distance] = image[center_x+i+1][center_y+distance]#右下
    return tmp

#往右或者左偏移一个像素
def shift(images,right=0,up=0):
    tmp=images.copy()
    for k in range(len(images)):
        for i in range(images[0].shape[0]):
            for j in range(images[0].shape[1]):
                if j+right>=28 or i+up>=28:
                    continue
                if j+right<0 or i+up<0:
                    continue
                tmp[k][i+up][j+right]=images[k][i][j]
    return tmp


def save_diff(images):
    tmp=images.copy()
    for k in range(len(images)):
        imagename="./save/"+str(k)+".jpg"
        plt.imsave(imagename,images[k].reshape(28,28),vmin=0, vmax=255,format="jpg",cmap='gray') 
        
    for k in range(len(images)):
        imagename="./save/"+str(k)+".jpg"
        image = Image.open(imagename, mode='r').convert('L')
        im_array = mp.pil_to_array(image)
        tmp[k]=im_array
    return tmp



#举一个例子，把输出都打印出来，做个示例计算一下，计算我们定义的距离
def example_rank(model,x_test,num):
    x_tmp = x_test.astype('float32').reshape(-1,28,28,1)
    x_tmp = x_tmp / 255
    bound_data_lst =[]
    out_index=len(model.layers)-1
    model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
  
    act_layers=model_layer.predict_on_batch(x_tmp)
    #notpasslist=find_notpass(model)
    #print 'act_layers:',len(act_layers)
    print act_layers[num]
    #order_lst,order_ratio_lst=boundary.order_output(act_layers)
    #dataframe = pd.DataFrame({'index':order_lst,'ratio':order_ratio_lst})
    #dataframe.to_csv('ratio_order_Example.csv') 
    return



def get_bound_data_mnist(model,x_test,y_test,bound_ratio=10):
    x_tmp = x_test.astype('float32').reshape(-1,28,28,1)
    x_tmp = x_tmp / 255
    bound_data_lst =[]
    out_index=len(model.layers)-1
    model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
  
    act_layers=model_layer.predict_on_batch(x_tmp)
    #notpasslist=find_notpass(model)
    #print 'act_layers:',len(act_layers)
    for i in range(len(act_layers)):#此i只是choice_index序化后
        act=act_layers[i]
        index,ratio = find_second(act)     
        if ratio< bound_ratio :
            bound_data_lst.append(i) 
                #print index,y_test[i]   
    x_bound = np.zeros((len(bound_data_lst),28,28))
    y_bound = np.zeros((len(bound_data_lst),))

    for i in range(len(bound_data_lst)):
        x_bound[i] = x_test[bound_data_lst[i]]
        y_bound[i] = y_test[bound_data_lst[i]]
    return x_bound,y_bound

def find_second(act):
    max_=0
    second_max=0
    index=0
    max_index=0
    for i in range(10):
        if act[i]>max_:
            max_=act[i]
            max_index=i
            
    for i in range(10):
        if i==max_index:
            continue
        if act[i]>second_max:#第2大加一个限制条件，那就是不能和max_一样
            second_max=act[i]
            index=i
    ratio=1.0*max_/second_max
    #print 'max:',max_index
    return index,ratio
'''
def diff(im_array1,im_array):
    for j in range(im_array.shape[0]):
        for k in range(im_array.shape[0]):
            if im_array[j][k]!=im_array1[j][k]:
                print im_array[j][k],im_array1[j][k]
    return

diff(im_array1,im_array)



'''
''' 
#读取图像
im = Image.open("lenna.jpg")
im.show()
 
# 指定逆时针旋转的角度
im_rotate = im.rotate(45) 
im_rotate.show()
'''
'''
def test():
    testjpg=np.zeros((28,28)) 
    testjpg[0][14]=255
    #testjpg[12][12]=2
    imagename="test_255.png"
    plt.imsave(imagename,testjpg,format="png",cmap='gray')
    return

test()

test1=plt.imread('test_1.png')
test255=plt.imread('test_255.png')

test1 = matplotlib.image.pil_to_array('test_1.png')
#test_2dot=plt.imread('test_2dot.jpg')

#plt.imshow(test_2dot, cmap='Greys_r')
#plt.show()

for i in range(28):
    for j in range(28):
        for k in range(3):
            if test1[i][j][k]!=test255[i][j][k]:
                print test1[i][j][k],test255[i][j][k]
                print i,j,k
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
'''
npzfile=np.load('mnist.npz') 
y_test= npzfile['y_test']
x_test= npzfile['x_test']  
x_train= npzfile['x_train']
y_train= npzfile['y_train']  
'''

model_path='../ModelA_raw.hdf5'
model=load_model(model_path)

x_tmp = x_test.astype('float32').reshape(-1,28,28,1)
x_tmp = x_tmp / 255

origin_passlist = find_pass(model,x_tmp,y_test)

y_tmp = keras.utils.to_categorical(y_test, 10)
    
score = model.evaluate(x_tmp, y_tmp)
#print('Test Loss: %.4f' % score[0])
print('Test case accuracy: %.4f'% score[1])

x_train = x_train.astype('float32').reshape(-1,28,28,1)
x_train = x_train / 255
y_train = keras.utils.to_categorical(y_train, 10)


score = model.evaluate(x_train, y_train)
#print('Test Loss: %.4f' % score[0])
print('Train case accuracy: %.4f'% score[1])

x_save = x_test.copy()
#x_test = darker(x_test)
x_test = rotate(x_test)
x_test = shift(x_test,0,-2)
#先旋转，后往右移2个事0.9003

#for i in range(len(x_test)):
        #x_axis = 13
        #random.randint(1, 26)
        #Return random integer in range [a, b]
        #y_axis = 13
        ##随机产生一个中心位置，然后把周围9个格子涂上全白
        #0黑,255是白
        #x_test[i][x_axis][y_axis]=255
        #x_test= add_9_white_dot(x_test,i,x_axis,y_axis)
        #加9宫格噪声，在中心位置是0.9042
        #加9宫格噪声，随机位置加是0.9587
        #只加一个点的噪音，0.9731
        #x_test = subsample(x_test,i)
        #下采样的准确率是0.97
        #x_test[i] = darker(x_test[i])


np.savez('beta_mnist_rotate(down2).npz',x_test=x_test,y_test=y_test)
#前面是名，后面是数组
#后缀npz加不加都一样

'''
npzfile=np.load('beta_mnist_shift_rotate(shift1left).npz') 
y_beta= npzfile['y_test']
x_beta= npzfile['x_test']  
'''
y_beta=y_test
x_beta=x_test

x_beta_tmp = x_beta.astype('float32').reshape(-1,28,28,1)
x_beta_tmp = x_beta_tmp / 255
mutated_passlist = find_pass(model,x_beta_tmp,y_beta)
y_beta_tmp = keras.utils.to_categorical(y_beta, 10)

#mutated_passlist = find_pass(model,x_beta,y_beta)

score = model.evaluate(x_beta_tmp, y_beta_tmp)
#print('Test Loss: %.4f' % score[0])
print('Test accuracy: %.4f'% score[1])

affected_list=[]
for i in origin_passlist:
    if i not in mutated_passlist:
        affected_list.append(i)
        
saveimage(x_beta,affected_list,y_test)
saveimage_origin(x_save,affected_list,y_test)

'''

image = Image.open('test.jpg', mode='r').convert('L')
im_array = mp.pil_to_array(image)
'''