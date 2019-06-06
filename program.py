#coding:utf-8

from glob import glob
import cv2
import sys
import re
from numpy import *
from skimage.feature import hog
import matplotlib.pylab as plt
from svmutil import *

from sklearn import svm
from sklearn import preprocessing

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
digits = "0123456789"
#########################################################3
def get_backgroup(path):
    """get backgroup"""
    
    labels=[]
    filenames=[]
    
    files = glob(path + "/*.jpg")
    files.sort()
    for f in files:
        filenames.append(f)
        labels.append(-1)
        
    return labels,filenames

def get_images(*path):
    """get image label to path """

    labels=[]
    filenames=[]

    for fname in path:
        path=fname[0:-11]
        with open(fname,"r") as txt:
            lines=txt.readlines()            
            for line in lines:
                if line=="\n":
                    continue
                t=line.split(" ")
                labels.append(t[1].strip())
                filenames.append(path+"/"+t[0])
                
    return labels,filenames

def get_Hog_Features(filenames):
    """get Features"""
    
    trainningMat=zeros((len(filenames),324))
    
    for index,f in enumerate(filenames):
        if not f.endswith("pgm"):
            if not f.endswith("jpg"):
                continue        
            
        img = cv2.imread(f, 0) # read image
        imgR = cv2.resize(img,(32, 32)) # normalize images to hog describer
        hist = hog(imgR, block_norm='L2-Hys') # hog describer     
        
        trainningMat[index,:]=hist
        
    return trainningMat        

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def mat2svmDate(dataSet,Labels):        
    n,m=dataSet.shape    
    print(m,n,len(Labels))
    with open("svmData.txt","a+") as W:
        x=0
        for i in range(n):
            W.write(str(Labels[i])+"\t")
            for j in range(m):
                if str(dataSet[i][j])==0:
                    continue
                W.write(str(j+1)+":"+str(dataSet[i][j])+"\t")
            if(x==n-1):
                break
            W.write("\r\n")
            x=x+1
                        
def train_data():
    labels,filenames= get_images("/home/cc/下载/ufl_scene_text_recognition-master/train/data/characters/icdar/img_ICDAR_train_labels.txt",
                                 "/home/cc/下载/ufl_scene_text_recognition-master/train/data/characters/synthetic/img_labels.txt",
                                 "/home/cc/下载/ufl_scene_text_recognition-master/train/data/characters/chars74k/img_labels.txt")

    backlabels,backfilenames=get_backgroup("/home/cc/下载/ufl_scene_text_recognition-master/train/data/background/train")    
    
    labels.extend(backlabels) 
    filenames.extend(backfilenames);   
    
    TrainMat= get_Hog_Features(filenames)
    #NorTrainMat= autoNorm(TrainMat)
    mat2svmDate(TrainMat,labels)
    #testlabels,testfilenames=get_images("/home/cc/下载/ufl_scene_text_recognition-master/train/data/characters/icdar/img_ICDAR_test_labels.txt")
    
    #kw={"dataArr":get_Hog_Features(filenames),"labelArr":labels,"testdataArr":get_Hog_Features(testfilenames),"testlabelArr":testlabels}
    
    #testDigits(get_Hog_Features(filenames),labels,get_Hog_Features(testfilenames),testlabels)


def make_scale_data(path):
    """修复特征文件"""
    with open(path,"r+") as T:
        lines=T.readlines()
        ##print(lines[1215])
        ##print(lines[1216])
        for index,f in enumerate(lines):    
            if(len( f.split(" "))!=1072):
                print(index)

from random import randint                    
def xrang_data(path):
    with open(path,"r") as T:
        lines=T.readlines()
        i=0
        with open("rand_svmData.svm","w") as P:
            while(i<5000):
                j=randint(0,len(lines))
                P.write(lines[j])
                i=i+1                                            
        
def train(): 
    y, x = svm_read_problem('svmData.scale')
    #param = svm_parameter('-c 8.0 -g 0.125 -m 4000 -h 1 -b 1')
    model = svm_train(y,x, '-c 8.0 -g 0.125 -m 4000 -h 1 -b 1')
    svm_save_model("svmData.model", model)
        
def test():
    model=svm_load_model("svmData.model")
    y, x = svm_read_problem('svmData3.scale')
    label,acc,val=svm_predict(y,x,model,'-b 1')
    #print(label)
    print(acc)
    
def get_range(path):
    with open(path,'r') as T:
        lines=T.readlines()
        return lines[2:]
    
def test_one(path):    
    img = cv2.imread(path, 0) # read image
    imgR = cv2.resize(img,(32, 32)) # normalize images to hog describer
    hist = hog(imgR, block_norm='L2-Hys') # hog describer 
    rang=get_range("svmData.range")
    for i in range(len(hist)):
        x=hist[i]
        y=float(rang[i].split(" ")[2].strip())
        hist[i]=round(x/y,7)
    #print(hist.shape)
    #hist=hist.reshape(2,-1)
    #print(hist.shape)
    #min_max_scaler=preprocessing.MinMaxScaler()
    #x_train_minmax=min_max_scaler.fit_transform(hist)
    #print(x_train_minmax.shape)
    #x_train_minmax=x_train_minmax.reshape(-1)
    #print(x_train_minmax.shape)
    mat_={}
    for index,d in enumerate(hist):
        mat_[index+1]=d
    #model=svm_load_model("svmData.model")
    
    y, x =[58],[mat_]
    label,acc,val=svm_predict(y,x,model,'-b 1')
    S=list(letters+digits)
    print(label) 
    if(label[0]==-1.0):
        return
    print(S[int(label[0])-1])

import argparse
import cv2

model=svm_load_model("svmData.model")

if __name__=="__main__":
    #test_one("data/characters/icdar/img_ICDAR_test/img_12.pgm")
    #test()
    #train()   
    #parser=argparse.ArgumentParser()
    #parser.add_argument("-i","--image",help="image path")
    #args=parser.parse_args()    
    while(True):
        im=input("input theimage path:")
        if im :
            
            test_one(im.strip())
            img= cv2.imread(im.strip())
            cv2.imshow("img",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        