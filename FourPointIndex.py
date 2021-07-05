# -*- coding: utf-8 -*-
"""
Spyder Editor
written by Eden & Effi.
This function will find the indexes of the areas of interest from a certain 4p test "data[start_index:end_index]",
and can be called externaly. This func. will first find the lower gradients areas, and remove a safety boundries 
from them.
inputs:
    data        = the acc. signal of the tested units. This is a 1D array of the entire signal.
    start_index = the starting index indicating where the 4p test begins. An integetr value. 
    end_index   = the ending index indicating where the 4p test ends. An integetr value.
    orientation      = 0 or 1.
    cont        = 1 for continious Tcyc only, every other test sholud get 0.
outputs:
    FPstartEnd  = A list of 4 lists, each contains the start & end indexes of each area of interest(A0,A90,A180,A270).
    Allfpa      = results measurements of every 4p
    
"""

def FourPointCalc(data, start_index,orientation , cont):

    
    import numpy as np
    from sklearn.cluster import KMeans
    from scipy.signal import find_peaks
    from scipy.signal import peak_widths
    from sklearn.neighbors import LocalOutlierFactor
    import statistics
    
    dataFpoint = data[:,0]
    
    # finding peaks indexes and peaks width:
    d=np.gradient(dataFpoint)
    peakp,_ =find_peaks(d,height=0.25*max(d))
    peakn,_ = find_peaks(-d,height=0.25*max(d))
    peaks = list(peakp) + list(peakn)
    peaks.sort()
    p2p=peak_widths(x=d,peaks=peaks,rel_height=0.25)
    p2n=peak_widths(x=-d,peaks=peaks,rel_height=0.25)
    peakWidthMax=int(np.ceil(max(max(p2p[0]),max(p2n[0])))*4)
    
    Gradients =d
    if(orientation):
        Gradients=Gradients[:peaks[-1]] 
    gradFPIndexes = np.where(abs(Gradients)<0.02*max(abs(Gradients)))[0]
    
    #partition indexes to a0,a90,a180,a270 areas
    kmeans = KMeans(init = 'k-means++',n_clusters = 4)
    y_kmeans = kmeans.fit_predict(gradFPIndexes.reshape(-1,1)) 
        
    FPstartEnd = []
    for i in range(4):
        temp = gradFPIndexes[np.where(y_kmeans==i)]
        #filtering outliers:
        clf = LocalOutlierFactor(n_neighbors=20)
        FPOutLiers= clf.fit_predict(temp.reshape(-1, 1))
        temp = temp[np.where(FPOutLiers==1)]
        FPstartEnd.append([temp[peakWidthMax],temp[-peakWidthMax]])
    FPstartEnd.sort()
    
    A1 = statistics.mean(dataFpoint[FPstartEnd[0][0]:FPstartEnd[0][1]])
    A2 = statistics.mean(dataFpoint[FPstartEnd[1][0]:FPstartEnd[1][1]])
    A3 = statistics.mean(dataFpoint[FPstartEnd[2][0]:FPstartEnd[2][1]])
    A4 = statistics.mean(dataFpoint[FPstartEnd[3][0]:FPstartEnd[3][1]])
    
    # calculate 4point params: SF in [mbit/g]; Bias in [g]; MA in [mrad]
    if(cont): #Tcyc cont
            if(orientation):
                SF=1000*(A1-A3)/2
                b=1000*(A4+A2)/2/SF
                ma=1000000*(A2-A4)/2/SF
            else: ######Need to be modified 
                SF=1000*(A1-A3)/2
                b=1000*(A4+A2)/2/SF
                ma=1000000*(A2-A4)/2/SF
    else: #Tcyc steps/STS/regular 4p
            if(orientation):
                SF=1000*(A3-A4)/2
                b=1000*(A1+A2)/2/SF
                ma=1000000*(A1-A2)/2/SF
            else:
                SF=1000*(A1-A2)/2
                b=1000*(A3+A4)/2/SF
                ma=1000000*(A4-A3)/2/SF
    
    TempSense = statistics.mean(data[:,1]) # [V[]]            
    allfpa = [SF,b,ma,TempSense] # [mbit/g]; [g]; [mrad]; [V]
    FPstartEnd = list(np.asarray(FPstartEnd)+start_index) 
    
    return FPstartEnd,allfpa



   
    
    
    