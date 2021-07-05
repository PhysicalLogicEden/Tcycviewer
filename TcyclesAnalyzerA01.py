# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:19:24 2021
"""
def TcyclesAnalyzer(filename,filepath):
    #%%Import libs
    #import external libs
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Cursor
    import numpy as np
    from sklearn.cluster import KMeans
    #from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import LocalOutlierFactor
    import statistics
    from scipy.signal import find_peaks
    import math
    import re
    import shutil
    import time
    from scipy.signal import decimate
    from mpl_toolkits.axes_grid1 import host_subplot
    from copy import deepcopy
    
    # import PL own function from path
    import os
    #os.chdir('W:\Operation\Production\Production_tools_environment\PTE')
    from FourPointIndex import FourPointCalc
    from math_oper import smooth
    from stdout_sup import suppress_stdout_stderr
    #%% Load data
    devNameInd = filename.find('L1')
    deviceName  = filename[devNameInd:devNameInd+14]
    dataTcyc = pd.read_table(os.path.join(filepath, filename),header=None)
    #%% output inits (from main - temporary)
    temps = dict(High1=85,High2=85,Low=-40) #[C]
    temps = dict(High1=75,High2=75,Low=-54) #[C]
    external_Tsensor = dict(sensitivity=0.008,offset = 66)# [V/C] ; [C]
    external_poly = dict(SFp=[0,0,0,0,0,13],Bp = [0,0,0,0,0,0],MAp = [0,0,0,0,0,0])


    #%% init
    
    decimation_factor = 30
    f_sample = 5 #Hz
    poly_rank = 5 # polynomial rank
    
    if re.search("cyc_L",filename):
        cycle1 = 1
        label = 'cyc1_'
    if re.search("cyc_2_L",filename):
        cycle1 = 0
        label = 'cyc2_'
    
    # loc_sep = [m.start() for m in re.finditer('/', filename)]
    # directoy_path = filename[:loc_sep[-1]]
    directoy_path =filepath
    if cycle1:
        Folder_path = [directoy_path+'/Tcycle1_'+deviceName+'_py']  #'_py' for comparison purpose only
    else:
        Folder_path = [directoy_path+'/Tcycle2_'+deviceName+'_py']  #'_py' for comparison purpose only
    if os.path.exists(Folder_path[0]): # delete existing folder
        shutil.rmtree(Folder_path[0])
    
    os.mkdir(Folder_path[0])
    os.chdir(directoy_path)

    
    #%% preprocess data
    
    dataTcyc = dataTcyc.iloc[:,:2].values
    
    if 1: # trim beginning of Tcycle (clear junk data at turn ON)
        tempeVdiff = np.gradient(dataTcyc[:1000,1])
        tempeVtrim=np.where(abs(tempeVdiff)>0.004)
        
        if (tempeVtrim[0].size > 0):
            dataTcyc = dataTcyc[tempeVtrim[0][-1]+1:,:] # clear junk data
        
    #Tsensor_resolution = max(abs((np.gradient(dataTcyc[:1000,1]))))
    


    #%% Tcycle test analyzer
    """""
    Tcycle test analyzer - this part will check whether its cont/steps, number of 4p, orientation and 
    finds the start to end points of each 4p measurements.
    input: dataTcyc - Temp cycle data(acc,temp)
    outputs:
        startEnd4p - list of starting and ending indexes of each 4point
        indexesOf4Ps - all the indexes of the same clustered 4p
    """""
    #dataFrameTcyc = pd.DataFrame(data=dataTcyc, columns=["acc", "temp"])

    cont = False
    if('cont' in filename):
        cont = True
    
    delta = 10
    Neg4p = np.where(dataTcyc[:,0]<0.8*np.min(dataTcyc[:,0]))[0]#finds indexes of the -1g areas for all 4p
    
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
    
    #%% find 4points indexes for beginning and end of each 4point section
    
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
    
    #%% creates allfpa and indexes:
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
    print('Total of four points in Tcycle:',numOf4p)
    
    #%% Tsensor:
    
    tv = allfpa[:,-1]
    
    T_sesnitivity = (max(tv)-min(tv))/(temps["High1"]-temps["Low"]) # [V/C]
    T_offset = max(tv)/T_sesnitivity-temps["High1"] # [C]
    
    if cycle1: # generate Tsensor info
        Tsensor =dict(sensitivity=T_sesnitivity,offset=T_offset)
        np.save('Tsensor.npy',Tsensor)
    else: # use Tsensor info from first cycle
        external_Tsensor = np.load('Tsensor.npy',allow_pickle='TRUE').item()
        Tsensor = external_Tsensor.copy()
    
    print('Temperature sensor before calibration: Offset',round(Tsensor["offset"]),'[C] ; Sensitivty',round(1000*Tsensor["sensitivity"],1),'[mV/C]')
    
    tc = tv/Tsensor["sensitivity"]-Tsensor["offset"]
    allfpa = np.append(allfpa,tc[:,None],axis=1) # [mbit/g] [g] [mrad] [tV] [tC]
    
    #temp_v = dict(min_tv = min(tv),fp1_tv = tv[0],max_tv = max(tv))
    
    
    # Tsensor.update(sensitivity=T_sesnitivity,offset=T_offset)
    #read_dictionary = np.load('Tsensor.npy',allow_pickle='TRUE').item()
    
    #%% filtering 4p areas
    fpIndToFilter=[]
    dataTcycFiltered = dataTcyc
    for i in range(numOf4p):
        fpIndToFilter.extend(list(range(startEnd4p[i][0],startEnd4p[i][1])))
    dataTcycFiltered = np.delete(dataTcyc, np.array(fpIndToFilter),axis = 0)
    
    alldatag = 1000000*dataTcycFiltered[:,0]/np.mean(allfpa[:,0]) # [mg]
    alltempe = dataTcycFiltered[:,1] # [V]
    alltempec = alltempe/Tsensor["sensitivity"]-Tsensor["offset"] # [C]
    
    #%% spikes finder
    # =============================================================================
    # filteredGrad = np.gradient(dataTcycFiltered[:,0])
    # # filtering local outliers from dataTcycFiltered
    # clf = LocalOutlierFactor(n_neighbors=30)
    # outliersfilteredGrad = clf.fit_predict(filteredGrad.reshape(-1, 1))
    # dataTcycFiltered=dataTcycFiltered[np.where(outliersfilteredGrad==-1)]
    # =============================================================================
    
    #%% Poly calculations
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
    SFerr = dict(std_err = np.std(SFer),mean_err = np.mean(SFer),max_err = max(abs(SFer))) # [ppm]
    Berr = dict(std_err = np.std(Ber),mean_err = np.mean(Ber),max_err = max(abs(Ber))) # [ug]
    MAerr = dict(std_err = 1000*np.std(MAer),mean_err = 1000*np.mean(MAer),max_err = 1000*max(abs(MAer))) # [urad]
    
    print('Bias residual error STD is:',round(Berr["std_err"]),'[ug]')
    if not(cycle1):
        print('Bias residual mean error is:',round(Berr["mean_err"]),'[ug]')
    
    #%% Tcycle Bias Delta:
    
    Tcycle_delta_bias_err = Ber[-1] - Ber[0] # [ug]
    Tcycle_p2p_bias_err = max(Ber) - min(Ber) # [ug]
    TcycleDeltaBias = dict(TDB_err = Tcycle_delta_bias_err,TDB_abs = abs(Tcycle_delta_bias_err),TDB_p2p = Tcycle_p2p_bias_err)
    print('Tcycle delta bias error is:',round(min(Ber)),'[ug]')
    
    #%% FULL temperature range sensitivities calculation (from poly):
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
    
    #%% FULL temperature range sensitivities calculation (between adjacent fpoints):
    
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
    print('Bias sensitivity:',round(MaxSensSteps["Bias"]),'[ug/C]')
    
    #%% saving results:      added at 20/06/2021 11:20
    if cycle1:
        Tcyc_Results = np.array([MaxSensSteps_SF,SFerr["std_err"],MaxSensSteps_Bias,Berr["std_err"],MaxSensSteps_MA,MAerr["std_err"],SFerr["mean_err"],Berr["mean_err"],MAerr["mean_err"],0,0,0,Tcycle_delta_bias_err]).reshape(1, -1)
        np.save('Tcyc_1_Results.npy',Tcyc_Results)
    else:
        Tcyc_Results = np.array([SFerr["std_err"],Berr["std_err"],MAerr["std_err"],SFerr["mean_err"],Berr["mean_err"],MAerr["mean_err"],Tcycle_delta_bias_err]).reshape(1, -1)
        np.save('Tcyc_2_Results.npy',Tcyc_Results)
    
    #%% smoothed data for plots:
    
    all_sf = np.polyval(SFp,alltempe) # [bit/g]
    all_bias = np.polyval(Bp,alltempe) # [g]
    datacomp = 1000*(1000*dataTcycFiltered[:,0]/all_sf - all_bias) # [mg] compensated
    
    #%% plots:
    
    if 0: # perform decimation for displayed alldata plots
        alldatag = decimate(alldatag,decimation_factor,20,'fir')
        alltempec = decimate(alltempec,decimation_factor,20,'fir')    
        datacomp = decimate(datacomp,decimation_factor,20,'fir')
    
    alldatag_s = smooth(alldatag,100) # [mg]
    alltempec_s = smooth(alltempec,100) # [C]
    datacompds = smooth(datacomp,100*decimation_factor) # [mg] compensated
    
    t = time.time()
    
    plot_folder = Folder_path[0]
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
    cursor = Cursor(host, useblit=True, color='red', linewidth=2)
    host.grid(color='lightgrey',linewidth=0.5)
    host.yaxis.get_label().set_color(p1.get_color())
    par.yaxis.get_label().set_color(p2.get_color())
    # leg = plt.legend(loc='best')
    # leg.texts[0].set_color(p1.get_color())
    # leg.texts[1].set_color(p2.get_color())
    w=host.figure.get_figwidth()
    h=host.figure.get_figheight()
    host.figure.set_size_inches(1.5*w,h*1.5)
    host.figure.savefig(os.path.join(plot_folder,'Tcyc_all_raw_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    # All Temp. Cycle data filtered w/o 4p
    host = host_subplot(111)
    par = host.twinx()
    p1, = host.plot(timeline_filt,alldatag_s,'royalblue',linewidth=0.5,label='Acceleration')
    p2, = par.plot(timeline_filt,alltempec_s,'coral',linewidth=0.5,label='Temperature')
    host.set_xlabel('Time [Hours]')
    host.set_ylabel('Acceleration [mg]')
    par.set_ylabel('Temperature [$^\circ$C]')
    plt.title('Temp. Cycle All Data w/o 4p - %s' %deviceName)
    cursor = Cursor(host, useblit=True, color='red', linewidth=2)
    host.grid(color='lightgrey',linewidth=0.5)
    host.yaxis.get_label().set_color(p1.get_color())
    par.yaxis.get_label().set_color(p2.get_color())
    # leg = plt.legend(loc='best')
    # leg.texts[0].set_color(p1.get_color())
    # leg.texts[1].set_color(p2.get_color())
    w=host.figure.get_figwidth()
    h=host.figure.get_figheight()
    host.figure.set_size_inches(1.5*w,h*1.5)
    host.figure.savefig(os.path.join(plot_folder,'Tcyc_all_data_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    # All Temp. Cycle Acc Vs Temp
    fig = plt.figure()
    plt.plot(alltempec_s,alldatag_s,'royalblue',linewidth=0.4)
    plt.title('Temp. Cycle Acc vs. Temp - %s'%deviceName)
    plt.xlabel('Temperature [$^\circ$C]')
    plt.ylabel('Output [mg]')
    plt.grid(color='lightgrey',linewidth=0.5)
    # plt.ylim([math.floor(min(alldatag_s)),math.ceil(1+max(alldatag_s))])
    # plt.yticks(np.arange(math.floor(min(alldatag_s)),math.ceil(1+max(alldatag_s)),1))
    w=fig.get_figwidth()
    h=fig.get_figheight()
    fig.set_size_inches(1.5*w,h*1.5)
    fig.savefig(os.path.join(plot_folder,'Tcyc_all_vs_temp_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    # Tcycle All Bias
    if cycle1: # use own poly
        fig = plt.figure()
        plt.plot(allfpa[:,4],1000*allfpa[:,1],'royalblue',linewidth=0.5,label='Temp. Cycle')
        plt.plot(allfpa[:,4],1000*Bcom,'magenta',dashes=[8, 4],linewidth=0.5,label='Poly. order: %d' %poly_rank)
    else: # use external poly
        if 0: # delete, for debug while creating script
            external_allfpa = deepcopy(allfpa)
            external_allfpa[:,1] = external_allfpa[:,1]+2/1000
        fig = plt.figure()
        plt.plot(external_allfpa[:,4],1000*external_allfpa[:,1],'royalblue',linewidth=0.5,label='Temp. Cycle 1')
        plt.plot(allfpa[:,4],1000*allfpa[:,1],'green',linewidth=0.5,label='Temp. Cycle 2')
        plt.plot(allfpa[:,4],1000*Bcom,'magenta',dashes=[8, 4],linewidth=0.5,label='Poly. order: %d' %poly_rank)
    plt.title('Temp. Cycle All Bias - %s'%deviceName)
    plt.xlabel('Temperature [$^\circ$C]')
    plt.ylabel('Bias [mg]')
    plt.grid(color='lightgrey',linewidth=0.5)
    leg = plt.legend(loc='best')
    w=fig.get_figwidth()
    h=fig.get_figheight()
    fig.set_size_inches(1.5*w,h*1.5)
    fig.savefig(os.path.join(plot_folder,'Tcyc_Bias_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.plot(allfpa[0,4],1000*allfpa[0,1],'r.-',linewidth=0.5,label='')
    plt.close()
    # plt.show()
    
    # Tcycle All SF
    if cycle1: # use own poly
        fig = plt.figure()
        plt.plot(allfpa[:,4],allfpa[:,0],'royalblue',linewidth=0.5,label='Temp. Cycle')
        plt.plot(allfpa[:,4],SFcom,'magenta',dashes=[8, 4],linewidth=0.5,label='Poly. order: %d' %poly_rank)
    else: # use external poly
        if 0: # delete, for debug while creating script
            external_allfpa = deepcopy(allfpa)
            external_allfpa[:,0] = external_allfpa[:,0]+0.01
        fig = plt.figure()
        plt.plot(external_allfpa[:,4],external_allfpa[:,0],'royalblue',linewidth=0.5,label='Temp. Cycle 1')
        plt.plot(allfpa[:,4],allfpa[:,0],'green',linewidth=0.5,label='Temp. Cycle 2')
        plt.plot(allfpa[:,4],SFcom,'magenta',dashes=[8, 4],linewidth=0.5,label='Poly. order: %d' %poly_rank)
    plt.title('Temp. Cycle All SF - %s'%deviceName)
    plt.xlabel('Temperature [$^\circ$C]')
    plt.ylabel('SF [0.5mLSB/g]')
    plt.grid(color='lightgrey',linewidth=0.5)
    leg = plt.legend(loc='best')
    w=fig.get_figwidth()
    h=fig.get_figheight()
    fig.set_size_inches(1.5*w,h*1.5)
    fig.savefig(os.path.join(plot_folder,'Tcyc_SF_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    # Tcycle All MA
    if cycle1: # use own poly
        fig = plt.figure()
        plt.plot(allfpa[:,4],allfpa[:,2],'royalblue',linewidth=0.5,label='Temp. Cycle')
        plt.plot(allfpa[:,4],MAcom,'magenta',dashes=[8, 4],linewidth=0.5,label='Poly. order: %d' %poly_rank)
    else: # use external poly
        if 0: # delete, for debug while creating script
            external_allfpa = deepcopy(allfpa)
            external_allfpa[:,2] = external_allfpa[:,2]+0.2
        fig = plt.figure()
        plt.plot(external_allfpa[:,4],external_allfpa[:,2],'royalblue',linewidth=0.5,label='Temp. Cycle 1')
        plt.plot(allfpa[:,4],allfpa[:,2],'green',linewidth=0.5,label='Temp. Cycle 2')
        plt.plot(allfpa[:,4],MAcom,'magenta',dashes=[8, 4],linewidth=0.5,label='Poly. order: %d' %poly_rank)
    plt.title('Temp. Cycle All MA - %s'%deviceName)
    plt.xlabel('Temperature [$^\circ$C]')
    plt.ylabel('MA [mrad]')
    plt.grid(color='lightgrey',linewidth=0.5)
    leg = plt.legend(loc='best')
    w=fig.get_figwidth()
    h=fig.get_figheight()
    fig.set_size_inches(1.5*w,h*1.5)
    fig.savefig(os.path.join(plot_folder,'Tcyc_MA_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    # SF error Vs Temp
    fig = plt.figure()
    plt.plot(allfpa[:,4],SFer,'royalblue',linewidth=0.4)
    plt.title('Temp. Cycle SF Error - %s'%deviceName)
    plt.xlabel('Temperature [$^\circ$C]')
    plt.ylabel('SF Error [ppm]')
    plt.grid(color='lightgrey',linewidth=0.5)
    # plt.ylim([math.floor(min(alldatag_s)),math.ceil(1+max(alldatag_s))])
    # plt.yticks(np.arange(math.floor(min(alldatag_s)),math.ceil(1+max(alldatag_s)),1))
    w=fig.get_figwidth()
    h=fig.get_figheight()
    fig.set_size_inches(1.5*w,h*1.5)
    fig.savefig(os.path.join(plot_folder,'Tcyc_ER_SF_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    # Bias error Vs Temp
    fig = plt.figure()
    plt.plot(allfpa[:,4],Ber,'royalblue',linewidth=0.4)
    plt.title('Temp. Cycle Bias Error - %s'%deviceName)
    plt.xlabel('Temperature [$^\circ$C]')
    plt.ylabel('Bias Error [$\mu$g]')
    plt.grid(color='lightgrey',linewidth=0.5)
    # plt.ylim([math.floor(min(alldatag_s)),math.ceil(1+max(alldatag_s))])
    # plt.yticks(np.arange(math.floor(min(alldatag_s)),math.ceil(1+max(alldatag_s)),1))
    w=fig.get_figwidth()
    h=fig.get_figheight()
    fig.set_size_inches(1.5*w,h*1.5)
    fig.savefig(os.path.join(plot_folder,'Tcyc_ER_Bias_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    # MA error Vs Temp
    fig = plt.figure()
    plt.plot(allfpa[:,4],1000*MAer,'royalblue',linewidth=0.4)
    plt.title('Temp. Cycle MA Error - %s'%deviceName)
    plt.xlabel('Temperature [$^\circ$C]')
    plt.ylabel('MA Error [$\mu$rad]')
    plt.grid(color='lightgrey',linewidth=0.5)
    # plt.ylim([math.floor(min(alldatag_s)),math.ceil(1+max(alldatag_s))])
    # plt.yticks(np.arange(math.floor(min(alldatag_s)),math.ceil(1+max(alldatag_s)),1))
    w=fig.get_figwidth()
    h=fig.get_figheight()
    fig.set_size_inches(1.5*w,h*1.5)
    fig.savefig(os.path.join(plot_folder,'Tcyc_ER_MA_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    # All Temp. Cycle data filtered w/o 4p - compensated
    host = host_subplot(111)
    par = host.twinx()
    p1, = host.plot(timeline_filt,datacompds,'royalblue',linewidth=0.5,label='Acceleration')
    p2, = par.plot(timeline_filt,alltempec_s,'coral',linewidth=0.5,label='Temperature')
    host.set_xlabel('Time [Hours]')
    host.set_ylabel('Acceleration [mg]')
    par.set_ylabel('Temperature [$^\circ$C]')
    plt.title('Temp. Cycle Acceleration (Compensated) - %s' %deviceName)
    cursor = Cursor(host, useblit=True, color='red', linewidth=2)
    host.grid(color='lightgrey',linewidth=0.5)
    host.yaxis.get_label().set_color(p1.get_color())
    par.yaxis.get_label().set_color(p2.get_color())
    # leg = plt.legend(loc='best')
    # leg.texts[0].set_color(p1.get_color())
    # leg.texts[1].set_color(p2.get_color())
    w=host.figure.get_figwidth()
    h=host.figure.get_figheight()
    host.figure.set_size_inches(1.5*w,h*1.5)
    host.figure.savefig(os.path.join(plot_folder,'Tcyc_data_comp_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    # All Temp. Cycle data filtered w/o 4p - compensated residual
    host = host_subplot(111)
    par = host.twinx()
    p1, = host.plot(timeline_filt,1000*(datacompds-statistics.mean(datacompds)),'royalblue',linewidth=0.5,label='Acceleration')
    p2, = par.plot(timeline_filt,alltempec_s,'coral',linewidth=0.5,label='Temperature')
    host.set_xlabel('Time [Hours]')
    host.set_ylabel('Acceleration Residual Error [$\mu$g]')
    par.set_ylabel('Temperature [$^\circ$C]')
    plt.title('Temp. Cycle Acceleration Residual Error - %s' %deviceName)
    cursor = Cursor(host, useblit=True, color='red', linewidth=2)
    host.grid(color='lightgrey',linewidth=0.5)
    host.yaxis.get_label().set_color(p1.get_color())
    par.yaxis.get_label().set_color(p2.get_color())
    # leg = plt.legend(loc='best')
    # leg.texts[0].set_color(p1.get_color())
    # leg.texts[1].set_color(p2.get_color())
    w=host.figure.get_figwidth()
    h=host.figure.get_figheight()
    host.figure.set_size_inches(1.5*w,h*1.5)
    host.figure.savefig(os.path.join(plot_folder,'Tcyc_data_res_%s%s.png'%(label,deviceName)),transparent=True,dpi=200)
    plt.close()
    # plt.show()
    
    elpse2 = time.time()-t
    print('plots time:',round(elpse2,2),'[sec]')
    
    #%%Output res
    return plot_folder

