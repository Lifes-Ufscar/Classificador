# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import pickle

dfeeg = pd.read_excel('Amostras//amostra_ID1_VD2_UI4.xlsx', sheet_name = 'Sheet1')

#####midGamma
lda_midGamma = pickle.load(open('lda_eeg//lda_midGamma.sav', 'rb'))
#####Final midGamma

#####lowGamma
lda_lowGamma = pickle.load(open('lda_eeg//lowGamma.sav', 'rb'))
#####Final lowGamma

#####highBeta
lda_highBeta = pickle.load(open('lda_eeg//highBeta.sav', 'rb'))
#####Final highBeta

#####lowBeta
lda_lowBeta = pickle.load(open('lda_eeg//lowBeta.sav', 'rb'))
#####Final lowBeta

#####highAlpha
lda_highAlpha = pickle.load(open('lda_eeg//highAlpha.sav', 'rb'))
#####Final highAlpha

#####lowAlphaa
lda_lowAlphaa = pickle.load(open('lda_eeg//lowAlphaa.sav', 'rb'))
#####Final lowAlphaa

#####theta
lda_theta = pickle.load(open('lda_eeg//theta.sav', 'rb'))
#####Final theta

#####delta
lda_delta = pickle.load(open('lda_eeg//delta.sav', 'rb'))
#####Final delta

#####delta
classificado_svm = pickle.load(open('lda_eeg//svm_classificador.sav', 'rb'))
#####Final delta





sinaldeltat = dfeeg.iloc[10:11, 2:]

tamanhoamostra = sinaldeltat.size

dfeeg = dfeeg.iloc[:, 2:]

aa = 2000
i = 0
#df1 = pd.DataFrame()
print("\nTESTE EEG INICIADO\n")
while aa <= tamanhoamostra:
   
    #Delta
    midGamma = dfeeg.iloc[10:11, i:aa]
    midGamma = midGamma.values
    midGamma = midGamma.astype(float)
    
    df1 = lda_midGamma.transform(midGamma)
    
    df1 = pd.DataFrame(df1)
   
    #lowGamma
    lowGamma = dfeeg.iloc[9:10, i:aa]
    lowGamma = lowGamma.values
    lowGamma = lowGamma.astype(float)
    
    lowGammalda = lda_lowGamma.transform(lowGamma)
    
    lowGammalda = pd.DataFrame(lowGammalda)
    x = lowGammalda.iloc[:, 0]
    x = pd.DataFrame(x)
    y = lowGammalda.iloc[:, 1]
    y = pd.DataFrame(y)

    df1['LG0'] = x
    df1['LG1'] = y
    
    #highBeta
    highBeta = dfeeg.iloc[8:9, i:aa]
    highBeta = highBeta.values
    highBeta = highBeta.astype(float)
    
    highBetalda = lda_highBeta.transform(highBeta)
    
    highBetalda = pd.DataFrame(highBetalda)
    x = highBetalda.iloc[:, 0]
    x = pd.DataFrame(x)
    y = highBetalda.iloc[:, 1]
    y = pd.DataFrame(y)

    df1['HB0'] = x
    df1['HB1'] = y
    
    #lowBeta
    lowBeta = dfeeg.iloc[7:8, i:aa]
    lowBeta = lowBeta.values
    lowBeta = lowBeta.astype(float)
    
    lowBetalda = lda_lowBeta.transform(lowBeta)
    
    lowBetalda = pd.DataFrame(lowBetalda)
    x = lowBetalda.iloc[:, 0]
    x = pd.DataFrame(x)
    y = lowBetalda.iloc[:, 1]
    y = pd.DataFrame(y)

    df1['LB0'] = x
    df1['LB1'] = y
    
    #highAlpha
    highAlpha = dfeeg.iloc[6:7, i:aa]
    highAlpha = highAlpha.values
    highAlpha = highAlpha.astype(float)
    
    highAlphalda = lda_highAlpha.transform(highAlpha)
    
    highAlphalda = pd.DataFrame(highAlphalda)
    x = highAlphalda.iloc[:, 0]
    x = pd.DataFrame(x)
    y = highAlphalda.iloc[:, 1]
    y = pd.DataFrame(y)

    df1['HA0'] = x
    df1['HA1'] = y
    
    #lowAlpha
    lowAlpha = dfeeg.iloc[5:6, i:aa]
    lowAlpha = lowAlpha.values
    lowAlpha = lowAlpha.astype(float)
    
    lowAlphalda = lda_lowAlphaa.transform(lowAlpha)
    
    lowAlphalda = pd.DataFrame(highAlphalda)
    x = lowAlphalda.iloc[:, 0]
    x = pd.DataFrame(x)
    y = lowAlphalda.iloc[:, 1]
    y = pd.DataFrame(y)

    df1['LA0'] = x
    df1['LA1'] = y
    
    #theta
    theta = dfeeg.iloc[4:5, i:aa]
    theta = theta.values
    theta = theta.astype(float)
    
    thetalda = lda_theta.transform(theta)
    
    thetalda = pd.DataFrame(thetalda)
    x = thetalda.iloc[:, 0]
    x = pd.DataFrame(x)
    y = thetalda.iloc[:, 1]
    y = pd.DataFrame(y)

    df1['T0'] = x
    df1['T1'] = y
    
    #delta
    delta = dfeeg.iloc[3:4, i:aa]
    delta = delta.values
    delta = delta.astype(float)
    
    deltalda = lda_delta.transform(delta)
    
    deltalda = pd.DataFrame(deltalda)
    x = deltalda.iloc[:, 0]
    x = pd.DataFrame(x)
    y = deltalda.iloc[:, 1]
    y = pd.DataFrame(y)

    df1['D0'] = x
    df1['D1'] = y
    
    df1 = df1.values
    
    predict = classificado_svm.predict(df1)
    
    
    
    
    
    
    
    print("predict", predict)
    
    i= i+200
    aa = aa+200
    total = 0
    
print("\nTESTE EEG FIM\n")
    





























