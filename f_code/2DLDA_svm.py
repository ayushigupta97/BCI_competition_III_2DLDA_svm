# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:22:53 2020

@author: AYUSHI GUPTA
"""

import functions as aks
from functions import *

f='D:/ceeri/f_code/'

sub=['aa','al','av','aw','ay']                                                              #five subjects
#sub=['ay'];
class_names = ["Right Hand","Foot"]                                                         #two motor imagery
n=5
acc_svm=np.zeros((5,n))
acc_lda=np.zeros((5,n))

bands=np.zeros((7,2));
fl=7
fh=9
for s in range(0,len(bands)):                                                               #Seven ranges of frequencies
    bands[s][0]=int(fl)
    bands[s][1]=int(fh)
    fl=fh
    fh=fh+2                                             
for j in range(0,len(sub)):                            
        cnt_subject= np.load(f+'subject_'+sub[j]+'_data.npy')
        cnt_subject_bandwise= np.load(f+'subject_'+sub[j]+'_data_bandwise.npy')
        xlabel= np.load(f+'subject_'+sub[j]+'_label.npy')
        train=np.load(f+'subject_'+sub[j]+'_train_index.npy')
        test=np.load(f+'subject_'+sub[j]+'_test_index.npy')                   
        print("subject is ", sub[j])  
        
        for i in range(5):
                train_index=train[i,:]
                test_index=test[i,:]
        
                train_lab_0=xlabel[train_index]                            
                train_lab_0=np.reshape(train_lab_0,(-1,len(train_lab_0)))                   #train labels
                test_lab_0=xlabel[test_index]
                test_lab_0=np.reshape(test_lab_0,(-1,len(test_lab_0)))                      #test labels
            
                traindata=[]
                testdata=[]
                for s in range(0,len(bands)):
                    cnt_subject1=(cnt_subject_bandwise[s,:,:,:])
                    r=[]
                    for k in range(len(cnt_subject1)):
                        l=cnt_subject[k].flatten()
                        r.append(l)
                    r=np.array(r)                                                           #converting 3d data to 2d data
                    cnt_subject1_train_0=cnt_subject1[train_index,:,:]                      #finding train data
                    cnt_subject1_test_0=cnt_subject1[test_index,:,:]                        #finding test data
                    r_train_0=r[train_index,:]
                    r_test_0=r[test_index,:]
                
                    U=aks.LDA2D(np.transpose(r_train_0),2,np.transpose(train_lab_0),118,300)     #finding feature matrix
                    U=np.real(U)
                    
                    u_dot_train=aks.dotpro(U,cnt_subject1_train_0)             #changing subdomain
                    u_dot_test=aks.dotpro(U,cnt_subject1_test_0)
                    u_train_0=aks.calmean(u_dot_train,1)
                    u_test_0=aks.calmean(u_dot_test,1)
                    traindata.append(u_train_0)                   
                    testdata.append(u_test_0)                   
                                  
                testdata=np.array(testdata)
                traindata=np.array(traindata)
                traind=np.squeeze(traindata[0,:,:])                            #train data
                testd=np.squeeze(testdata[0,:,:])                              #test data
                       
                for k in range(1,len(bands)):
                    traind=np.concatenate((traind,np.squeeze(traindata[k,:,:])),axis=1)
                    testd=np.concatenate((testd,np.squeeze(testdata[k,:,:])),axis=1)

                traind, testd=aks.my_featsel(traind, testd,np.transpose(train_lab_0),np.transpose(test_lab_0),10)            #feature selection
                clf=svm.SVC(kernel='rbf')                                      #support vector machine              
                clf.fit(traind,np.transpose(train_lab_0))
                y_pred=clf.predict(testd)
                acc_svm[j][i]=metrics.accuracy_score(np.transpose(test_lab_0), y_pred)
    
                print("Accuracy SVM:",acc_svm[j][i])
                plt.figure()                                                   #confusion matrix 
                plot_confusion_matrix(test_lab_0,y_pred,class_names,
                          title='Confusion matrix SVM subject '+sub[j],
                          cmap=plt.cm.Blues)    
                clf=LDA()                                                      #Linear Discriminant Analysis
                clf.fit(traind,np.transpose(train_lab_0))
                y_pred=clf.predict(testd)
                acc_lda[j][i]=metrics.accuracy_score(np.transpose(test_lab_0), y_pred)
                print("Accuracy LDA:", acc_lda[j][i])
                plt.figure()
                plot_confusion_matrix(test_lab_0,y_pred,class_names,           #confusion matrix
                          title='Confusion matrix LDA '+sub[j],
                          cmap=plt.cm.Greens)
                
        
            
        



    