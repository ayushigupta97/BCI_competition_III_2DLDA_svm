# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:22:53 2020

@author: AYUSHI GUPTA
"""


import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import signal
from itertools import count
from sklearn import svm
from sklearn import metrics
from itertools import count
from numpy.linalg import matrix_rank
from scipy.signal import butter, lfilter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import kurtosis as ku
from sklearn.metrics import confusion_matrix

#Covariance Matrix

def covariance(a):                                                             #covariance matrix
    t=[]
    for i in range(len(a)):
        x=np.cov(a[i])
        t.append(x)
    t=np.array(t)
    return t

#Two Dimentional LDA

def LDA2D(X,classNumber,xlabel,height,width):                                  #finding feature matrix
    eachclass=np.zeros((1,classNumber))
    [m,n]=np.shape(X)
    meanX=np.array(X.mean(1))
    meanX=meanX.reshape(height,width)
    sum0=np.zeros((height,height))
    for i in range(n):
        x1=X[:,i].reshape(height,width)                             
        sum0=np.add(sum0,np.cov(x1))
    sb=np.zeros((height,height))
    for i in range(classNumber):
        dd=[]
        for j in range(len(xlabel)):
            if(xlabel[j][0]==i):
                dd.append(j)
        dd=np.array(dd)
        dd2=np.transpose([dd])
        m1=len(dd2)
        #n1=len(dd2[0])
        eachclass[0][i]=m1
        X1=[]
        for j in range(len(dd)):
            t=[]
            for k in range(len(X)):
                t.append(X[k][j])
            a=np.array(t)
            X1.append(a)
        X1=np.array(X1)
        meanXX=X1.mean(0)
        meanXX=np.transpose(meanXX)
        XX1=meanXX.reshape((height,width))
        M1=np.squeeze([XX1-meanX])
        sb=sb+np.dot(m1,np.dot(M1,M1.transpose()))
    sum1=sum0+sb
    q=matrix_rank(sum1)
    invsS=np.linalg.inv(sum1)
    C=np.dot((invsS),sb)    
    eigval, eigvec = np.linalg.eig(C)
    a=np.argsort(eigval)[::-1]
    u=(eigvec[a,:])
    return u
    
    #Dot Product

def dotpro(a,b):                                                               #dotproduct matrix
    c=[]
    for i in range(len(b)):
        p=np.dot(a,b[i])
        c.append(p)
    c=np.array(c)
    return c

#Feature Vector
    
def calmean(a,k):                                                              #finding features
    t=[] 
    for i in range(len(a)):
        l=(np.squeeze(a[i,:,:]))        
        m1=l[:k,:]
        m2=l[-k-1:-1,:]       
        feat=np.vstack([m1,m2])
#        feat1=np.array(np.log(np.var(feat,axis=1)))
        feat1=np.array(np.var(feat,axis=1))
        feat2=np.array(np.mean(feat,axis=1))
        feat3=np.array(np.max(feat,axis=1))
        feat4=ku(feat,axis=1)
        f1=np.concatenate((feat1,feat2,feat3,feat4),axis=0)
        t.append(f1)
#        t=np.concatenate((feat1,feat2),axis=1)
    t=np.array(t) 
    return t

#Filtering the data
    
def butter_bandpass(lowcut, highcut, fs, order=4):                             #bandpass filter                  
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data,xlabel, lowcut, highcut, fs=100, order=4):
    for i in range (0,len(xlabel)):
        b, a = butter_bandpass(lowcut, highcut, fs=100, order=4)
        x=  np.squeeze(np.array(data[i,:,:]))
        data[i,:,:] = lfilter(b, a, x)
   
    return data
def my_featsel(a,b,c,d,n):                                                     #finding features 
    e=np.absolute(np.concatenate((a,b),axis=0))
    f=np.concatenate((c,d),axis=0)
    X_new = SelectKBest(chi2, k=n).fit_transform(e,f)
    a_new=X_new[0:len(a),:]
    b_new=X_new[len(a):(len(a)+len(b)),:]
    return a_new,b_new

def plot_confusion_matrix(test_lab_0,y_pred,classes,                           #plotting confusion
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm=confusion_matrix(np.transpose(test_lab_0),y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #print()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)