# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:07:57 2021

@author: eden
"""
#pip install streamlit

import streamlit as st
import pandas as pd
from pandas import DataFrame
# import PL own function from path
# from TcyclesAnalyzerA02 import TcyclesAnalyzer2
import os
import matplotlib.pyplot as plt
# from matplotlib.widgets import Cursor
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
# from tkinter import Tk     # from tkinter import Tk for Python 3.x
# from tkinter.filedialog import askopenfilename
import statistics as stat
from scipy.signal import find_peaks
import math
import re
# import shutil
import time
from scipy.signal import decimate
# import pickle
from mpl_toolkits.axes_grid1 import host_subplot
from copy import deepcopy

# import PL own function from path
import os
from FourPointIndex import FourPointCalc
from math_oper import smooth
from stdout_sup import suppress_stdout_stderr

Analyzed_folder_path = 'W:\\Operation\\Production\\Production_tools_environment\\PTE\\uploads'

st.title("Physical-Logic Temp cycle Analyzer")

st.write('please choose Tcyc.txt file')


 
    
uploaded_file = st.file_uploader("Upload File",type=['txt']) 
if uploaded_file is not None:
    file_details = {"Filename":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details)
    # Check File Type
    if uploaded_file.type == "text/plain":
        df = str(uploaded_file.read(),"utf-8")
        dataTcyc =pd.DataFrame([np.fromstring(x, dtype='float64', sep='\t') for x in df.split('\n')])
        filename = uploaded_file.name
        devNameInd = filename.find('L1')
        deviceName  = filename[devNameInd:devNameInd+14]
        #########output inits (from main - temporary)
        temps = dict(High1=85,High2=85,Low=-40) #[C]
        temps = dict(High1=75,High2=75,Low=-54) #[C]
        external_Tsensor = dict(sensitivity=0.008,offset = 66)# [V/C] ; [C]
        external_poly = dict(SFp=[0,0,0,0,0,13],Bp = [0,0,0,0,0,0],MAp = [0,0,0,0,0,0])
        #########init
        decimation_factor = 30
        f_sample = 5 #Hz
        poly_rank = 5 # polynomial rank
        if re.search("cyc_L",filename):
            cycle1 = 1
            label = 'cyc1_'
        if re.search("cyc_2_L",filename):
            cycle1 = 0
            label = 'cyc2_'
        #########preprocess data
        dataTcyc = dataTcyc.iloc[:,:2].values
        if 1: # trim beginning of Tcycle (clear junk data at turn ON)
            tempeVdiff = np.gradient(dataTcyc[:1000,1].astype(np.float))
            tempeVtrim=np.where(abs(tempeVdiff)>0.004)
            if (tempeVtrim[0].size > 0):
                dataTcyc = dataTcyc[tempeVtrim[0][-1]+1:,:] # clear junk data
        #########Tcycle test analyzer
        cont = False
        if('cont' in filename):
            cont = True
        delta = 10
        st.write(dataTcyc[:,0])
        st.write(np.minimum(dataTcyc[:,0]))
        Neg4p = np.where(dataTcyc[:,0]<0.8*np.minimum(dataTcyc[:,0]))[0]#finds indexes of the -1g areas for all 4p
            #filtering outliers:
        clf = LocalOutlierFactor(n_neighbors=10)
        Neg4pOutLiers= clf.fit_predict(Neg4p.reshape(-1, 1))
        Neg4p = Neg4p[np.where(Neg4pOutLiers==1)]
            #Number of 4p:
        GradNeg4p=np.gradient(Neg4p)
        peaks= find_peaks(GradNeg4p,distance=40) # distance should be smaller than 4point samples at -1g
        numOf4p = int(len(peaks[0]))+1
            #size of areas of interest:
        meanArea = int(np.ceil(np.mean(np.diff(peaks[0]))))
        #########find 4points indexes for beginning and end of each 4point section
        startEnd4p=[]
        if(cont):
            #clustering gradients indexes for each -1g area of 4p 
            kmeans = KMeans(init = 'k-means++',n_clusters = numOf4p)
            Neg4p_y_kmeans = kmeans.fit_predict(Neg4p.reshape(-1,1))
            #orientation:
            orientation =1
            for i in range(numOf4p):
                midNeg4p= math.ceil(np.median(Neg4p[np.where(Neg4p_y_kmeans==i)]))#middle of Neg4p
                startEnd4p.append([midNeg4p-int(meanArea*2.5)-2*delta,midNeg4p+int(1.5*meanArea)+4*delta])
        else:
            grad =np.gradient(dataTcyc[:,0])
            #clustering grad into 2 clusters to find indForCalc 
            kmeans = KMeans(init = 'k-means++',n_clusters = 2)
            indForCalc_y_kmeans = kmeans.fit_predict(abs(grad).reshape(-1,1)) 
            indForCalc = np.where(indForCalc_y_kmeans==1)[0] #first filter
            #clustering gradients indexes for each 4p 
            kmeans = KMeans(init = 'k-means++',n_clusters = numOf4p)
            y_kmeans = kmeans.fit_predict(indForCalc.reshape(-1,1)) 
            #orientation:
            orientation = int(len(find_peaks(grad[indForCalc[0]-500:indForCalc[0]+1500],height=0.75*max(abs(grad[indForCalc[0]:indForCalc[0]+1200])))[0])==3)  
            #LOF: Local Outliers Factor to find outliers areas with high gradients other than requested 20/06/2021
            clf = LocalOutlierFactor(n_neighbors=8)
            indexesOf4Ps = []
            for i in range(numOf4p):
                temp = indForCalc[np.where(y_kmeans==i)]
                outliersGradients = clf.fit_predict(temp.reshape(-1, 1))
                temp = temp[np.where(outliersGradients==1)]
                indexesOf4Ps.append(temp) 
                startEnd4p.append([indexesOf4Ps[i][0]-2*delta,indexesOf4Ps[i][-1]+2*delta])
        startEnd4p.sort()
        startEnd4p[-1][1] = min(startEnd4p[-1][1],len(dataTcyc))
        #########creates allfpa and indexes
        gradIndexes = []
        Indexes = []
        allfpa = []
        for i in range(numOf4p):
            with suppress_stdout_stderr():  # using the func. while avoiding stdout printing in console
                FourPointareas, allfpares = FourPointCalc(dataTcyc[startEnd4p[i][0]:startEnd4p[i][1]],startEnd4p[i][0],orientation,cont)
            Indexes.append([FourPointareas[0][0],FourPointareas[0][1],FourPointareas[1][0],FourPointareas[1][1],FourPointareas[2][0],FourPointareas[2][1],FourPointareas[3][0],FourPointareas[3][1]]) #stars&end points for all 4 areas of each 4p test
            #indexes of data to calculate 4p measurements - SEND TO IF 0
            gradIndexes.append(list(range(int(Indexes[i][0]),int(Indexes[i][1])))+list(range(int(Indexes[i][2]),int(Indexes[i][3])))+list(range(int(Indexes[i][4]),int(Indexes[i][5])))+list(range(int(Indexes[i][6]),int(Indexes[i][7])))) 
            allfpa.append(allfpares)
        allfpa=np.array(allfpa) # [mbit/g]; [g]; [mrad]; [V]
        st.write('Total of four points in Tcycle:',numOf4p)
        #########Tsensor
        tv = allfpa[:,-1]
        T_sesnitivity = (max(tv)-min(tv))/(temps["High1"]-temps["Low"]) # [V/C]
        T_offset = max(tv)/T_sesnitivity-temps["High1"] # [C]
        if cycle1: # generate Tsensor info
            Tsensor =dict(sensitivity=T_sesnitivity,offset=T_offset)
            np.save('Tsensor.npy',Tsensor)
        else: # use Tsensor info from first cycle
            external_Tsensor = np.load('Tsensor.npy',allow_pickle='TRUE').item()
            Tsensor = external_Tsensor.copy()
        st.write('Temperature sensor before calibration: Offset',round(Tsensor["offset"]),'[C] ; Sensitivty',round(1000*Tsensor["sensitivity"],1),'[mV/C]')
        tc = tv/Tsensor["sensitivity"]-Tsensor["offset"]
        allfpa = np.append(allfpa,tc[:,None],axis=1) # [mbit/g] [g] [mrad] [tV] [tC]
        #########filtering 4p areas
        fpIndToFilter=[]
        dataTcycFiltered = dataTcyc
        for i in range(numOf4p):
            fpIndToFilter.extend(list(range(startEnd4p[i][0],startEnd4p[i][1])))
        dataTcycFiltered = np.delete(dataTcyc, np.array(fpIndToFilter),axis = 0)
        alldatag = 1000000*dataTcycFiltered[:,0]/np.mean(allfpa[:,0]) # [mg]
        alltempe = dataTcycFiltered[:,1] # [V]
        alltempec = alltempe/Tsensor["sensitivity"]-Tsensor["offset"] # [C]
        ########Poly calculations
        SFp = np.polyfit(tv,allfpa[:,0],poly_rank) # SF poly
        Bp=np.polyfit(tv,allfpa[:,1],poly_rank) # Bias poly
        MAp=np.polyfit(tv,allfpa[:,2],poly_rank) # MA poly
        if cycle1: # save poly ana allfpa if it is first cycle
            poly = dict(SFp = SFp,Bp = Bp,MAp = MAp)
            np.save('poly.npy',poly)
            np.save('allfpa.npy',allfpa)
        else: # use poly from first cycle. load first cycle allfpa as well for reference
            external_allfpa = np.load('allfpa.npy')
            external_poly = np.load('poly.npy',allow_pickle='TRUE').item()
            poly = external_poly.copy()
            SFp = poly["SFp"]
            Bp = poly["Bp"]
            MAp = poly["MAp"]
        SFcom = np.polyval(SFp,tv) # SF compensating values [bit/g]
        SFer = 1000000*(allfpa[:,0]-SFcom)/allfpa[:,0] # SF error [ppm]
        Bcom = np.polyval(Bp,tv) # Bias compensating values [g]
        Ber = 1000000*(allfpa[:,1] - Bcom) # Bias error [ug]
        MAcom = np.polyval(MAp,tv) # MA compensating values
        MAer = 1*(allfpa[:,2] - MAcom) # MA error [mrad]
            # Residual error, mean error, max error:
        SFerr = dict(std_err = stat.stdev(SFer),mean_err = np.mean(SFer),max_err = max(abs(SFer))) # [ppm]
        Berr = dict(std_err = stat.stdev(Ber),mean_err = np.mean(Ber),max_err = max(abs(Ber))) # [ug]
        MAerr = dict(std_err = 1000*stat.stdev(MAer),mean_err = 1000*np.mean(MAer),max_err = 1000*max(abs(MAer))) # [urad]
        st.write('Bias residual error STD is:',round(Berr["std_err"]),'[ug]')
        if not(cycle1):
            st.write('Bias residual mean error is:',round(Berr["mean_err"]),'[ug]')
        ########Tcycle Bias Delta
        Tcycle_delta_bias_err = Ber[-1] - Ber[0] # [ug]
        Tcycle_p2p_bias_err = max(Ber) - min(Ber) # [ug]
        TcycleDeltaBias = dict(TDB_err = Tcycle_delta_bias_err,TDB_abs = abs(Tcycle_delta_bias_err),TDB_p2p = Tcycle_p2p_bias_err)
        st.write('Tcycle delta bias error is:',round(min(Ber)),'[ug]')
        ########FULL temperature range sensitivities calculation (from poly)
        if cycle1:
            tv2 = list(range(temps["Low"],temps["High1"]+1))
        else:
            tv2 = list(range(temps["Low"],temps["High2"]+1))
        tv2 = (tv2+Tsensor["offset"])*Tsensor["sensitivity"]
        SFbyPol = np.polyval(SFp,tv2) # [mbit/g]
        BiasbyPol = np.polyval(Bp,tv2) # [g]
        MAbyPol = np.polyval(MAp,tv2); # [mrad]
        SensPoly_SF = 1000000*np.diff(SFbyPol)/np.mean(allfpa[:,0]) # SF sensitivities [ppm/C]
        SensPoly_Bias = 1000000*np.diff(BiasbyPol) # Bias sensitivities [ug/C]
        SensPoly_MA = 1000*np.diff(MAbyPol) # MA sensitivities [urad/C]
        MaxSensPoly_SF = max(abs(SensPoly_SF)); # max SF sensitivity [ppm/C]
        MaxSensPoly_Bias = max(abs(SensPoly_Bias)); # max Bias sensitivity [ug/C]
        MaxSensPoly_MA = max(abs(SensPoly_MA)); # max MA  sensitivity [urad/C]
        MaxSensPoly = dict(SF = MaxSensPoly_SF,Bias = MaxSensPoly_Bias,MA = MaxSensPoly_MA)
        ########FULL temperature range sensitivities calculation (between adjacent fpoints)
        SensSteps_Temp = np.diff(allfpa[:,4]) # delta temp C between steps
        SensSteps_SF = 1000000*np.diff(allfpa[:,0])/(SensSteps_Temp)/np.mean(allfpa[:,0]) # SF sensitivities [ppm/C]
        SensSteps_Bias = 1000000*np.diff(allfpa[:,1])/(SensSteps_Temp) # Bias sensitivities [ug/C]
        SensSteps_MA = 1000*np.diff(allfpa[:,2])/(SensSteps_Temp) # MA sensitivities [urad/C]
        if cont: # Remove steps sensitivity calculation for Max Temp due to high temperature drift:
            max_T_step = np.where(allfpa[:,4] == max(allfpa[:,4]))[0][0]
            SensSteps_Temp = np.delete(SensSteps_Temp,max_T_step-1)
            SensSteps_SF = np.delete(SensSteps_SF,max_T_step-1)
            SensSteps_Bias = np.delete(SensSteps_Bias,max_T_step-1)
            SensSteps_MA = np.delete(SensSteps_MA,max_T_step-1)
            # Remove steps where temperature change is smaller than 1C, to avoid dividing by small value
        RemoveFromSteps = np.where(abs(SensSteps_Temp)<1)[0]
        SensSteps_Temp = np.delete(SensSteps_Temp,RemoveFromSteps)
        SensSteps_SF = np.delete(SensSteps_SF,RemoveFromSteps)
        SensSteps_Bias = np.delete(SensSteps_Bias,RemoveFromSteps)
        SensSteps_MA = np.delete(SensSteps_MA,RemoveFromSteps)   
        SensSteps = dict(Temp = SensSteps_Temp,SF = SensSteps_SF,Bias = SensSteps_Bias,MA = SensSteps_MA)
        MaxSensSteps_SF = max(abs(SensSteps["SF"])) # max SF sensitivity [ppm/C]
        MaxSensSteps_Bias = max(abs(SensSteps["Bias"])) # max Bias sensitivity [ug/C]
        MaxSensSteps_MA = max(abs(SensSteps["MA"])) # max MA  sensitivity [urad/C]
        MaxSensSteps = dict(SF = MaxSensSteps_SF,Bias = MaxSensSteps_Bias,MA = MaxSensSteps_MA)
        st.write('Bias sensitivity:',round(MaxSensSteps["Bias"]),'[ug/C]')
        ########smoothed data for plots
        all_sf = np.polyval(SFp,alltempe) # [bit/g]
        all_bias = np.polyval(Bp,alltempe) # [g]
        datacomp = 1000*(1000*dataTcycFiltered[:,0]/all_sf - all_bias) # [mg] compensated
        ########plots
        alldatag_s = smooth(alldatag,100) # [mg]
        alltempec_s = smooth(alltempec,100) # [C]
        datacompds = smooth(datacomp,100*decimation_factor) # [mg] compensated
        t = time.time()
        timeline_all = np.arange(len(dataTcyc[:,0]))/f_sample/60/60 #Hours
        timeline_filt = np.arange(len(alldatag))/f_sample/60/60 #Hours
        plt.rcParams['axes.formatter.useoffset'] = False
            # All Temp. Cycle raw data (acc and temp)
        host = host_subplot(111)
        par = host.twinx()
        p1, = host.plot(timeline_all,dataTcyc[:,0]/abs(allfpa[0,0])*1000000,'royalblue',linewidth=0.5,label='Acceleration')
        p2, = par.plot(timeline_all,dataTcyc[:,1],'coral',linewidth=0.5,label='Temperature')
        host.set_xlabel('Time [Hours]')
        host.set_ylabel('Acceleration [mg]')
        par.set_ylabel('Temperature [V]')
        plt.title('Temp. Cycle Raw Data - %s' %deviceName)
        host.grid(color='lightgrey',linewidth=0.5)
        host.yaxis.get_label().set_color(p1.get_color())
        par.yaxis.get_label().set_color(p2.get_color())
        # leg = plt.legend(loc='best')
        # leg.texts[0].set_color(p1.get_color())
        # leg.texts[1].set_color(p2.get_color())
        w=host.figure.get_figwidth()
        h=host.figure.get_figheight()
        host.figure.set_size_inches(1.5*w,h*1.5)
        st.pyplot(host)
        
        
        
        
        
        
        
