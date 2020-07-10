# BCI_competition_III_2DLDA_svm
Physical Activity Recognition Using Brain Signals

## **Motor Imagery Brain Computer Interface**
BCI is one of the field in the computer science which proves out to be boon for such people.Brain Computer Interface is the device through which the user can interact with computers with the help of brain signals only. Simulating the brain signals from motor cortex by imagining motor movements without any actual movement or stressing the muscles.

## **Electroencephalography(EEG)** 
EEG is a measuring tool used to record the electrical signals generated by the brain via electrodes placed on the scalp.It records electrical patterns of the brain. Electroencephalogram (EEG) based brain-computer interfaces (BCI) monitor neural activity and translate these signals into actions and/or decisions, with the final goal of enabling users to interact with a computer using only their thoughts.  

## **2 Dimensional-Linear Discriminant Analysis(Feature Extraction)**
LDA is a dimensionality reduction method. The classical LDA suffers from the singularity problem means that the algorithm fails when the all the scatter matrices are singular. To overcome this problem, here we introduce 2DLDA stands for 2-Dimensional Linear Discriminant Analysis. Classical LDA works with vectorised data whereas 2DLDA works with data in matrix form.

## **Feature Selection**
The process of selecting those features which contribute most to the output variable is referred to as Feature Selection. Suppose there are two events, then chi square test in statistics tells the independence of these two events. Chi-Square gives the relationship between Expected Count E and the Observed Count O.

## **Support Vector Machine(Classification Algorithm)**
The main objective of Support Vector Machine is to find the hyperplanes in N-dimensional space for N features that can distinctly classify data points. To separate the data points of two classes there can be as many hyperplanes are possible but the motive is to find hyperplanes which have maximum distance between data points of each classes with the plane. 

## **Dataset Used**
Dataset IVa from BCI Competition III is used http://www.bbci.de/competition/iii

## **Installation**
1. **Implementation** - Project implemented in python 2.7
2. **IDE** - spyder (Anaconda Python Distribution)
3. **Data-set Description**  
    1. Five Subjects **{aa, al, av, aw, ay}**
    2. 118 channels
    3. Two motor imagery **Right hand, Foot**
    4. 280 trials, 140 for each task.
    5. Downsampling Frequency 100 Hz
4. **Libraries Used**
    1. numpy
    2. scipy
    3. itertools
    4. sklearn
    5. matplotlib
