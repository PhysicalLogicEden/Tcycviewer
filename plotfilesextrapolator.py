# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:49:12 2021

@author: eden
"""
#%%Import libs
from tkinter import Tk     # from tkinter import Tk for Python 3.x
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import os
from glob import glob
from matplotlib.widgets import Cursor
#pip install seaborn
import seaborn as sns

#%% Choose location of folder
root = Tk()
root.attributes('-topmost', 1)
root.withdraw()
folder_selected = filedialog.askdirectory(parent=root)

#%% list of txt files in the location directory
txtAllFiles = [y for x in os.walk(folder_selected) for y in glob(os.path.join(x[0], '*.txt'))]
# choise params:
matchWord = 'Custom_L'    #regex to be found
colnum = 2            #0 - all data, 1 - acc, 2- temp
overlayed = True        #True - all in one plot

#%% data preprocessing
relevantDataFiles = []
for i in range(len(txtAllFiles)):
    if(matchWord in txtAllFiles[i]):
        relevantDataFiles.append(txtAllFiles[i])
if len(relevantDataFiles)>0:
    table = pd.read_table(relevantDataFiles[0],header=None)
    dataacc = table.iloc[:,0].values
    datatemp = table.iloc[:,1].values
    for i in range(1,len(relevantDataFiles)):
        table = pd.read_table(relevantDataFiles[i],header=None)
        acc = table.iloc[:len(datatemp),0].values
        temp = table.iloc[:len(datatemp),1].values
        datatemp = np.column_stack((datatemp,temp))
        dataacc = np.column_stack((dataacc,acc))
else:
    print('####### No %s files where found #######'%matchWord)

#%% plot according to choise params

fig = plt.figure(dpi=250)
ax = fig.subplots()
cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,color = 'r',linewidth=0.4)
if overlayed:
    timeline = np.linspace(0, 1, len(datatemp))
    if colnum==0:
        for i in range(len(relevantDataFiles)):
            ax.plot(timeline,dataacc[:,i],linewidth=0.4)
            ax.plot(timeline,datatemp[:,i],linewidth=0.4)
        plt.show()
    if colnum==1:
        for i in range(len(relevantDataFiles)):
            ax.plot(timeline,dataacc[:,i],linewidth=0.4)
        plt.show()
    if colnum==2:
        for i in range(len(relevantDataFiles)):
            ax.plot(timeline,datatemp[:,i],linewidth=0.4)
        plt.show()
    plt.grid(color='lightgrey',linewidth=0.5)
        
else:
    timeline = np.linspace(0, 1, len(datatemp))
    if colnum==0:
        for i in range(len(relevantDataFiles)):
            ax.plot(timeline,dataacc[:,i],linewidth=0.4)
            ax.plot(timeline,datatemp[:,i],linewidth=0.4)
        plt.show()
    if colnum==1:
        for i in range(len(relevantDataFiles)):
            ax.plot(timeline,dataacc[:,i],linewidth=0.4)
        plt.show()
    if colnum==2:
        for i in range(len(relevantDataFiles)):
            ax.plot(timeline,datatemp[:,i],linewidth=0.4)
        plt.show()
    plt.grid(color='lightgrey',linewidth=0.5)
    
    
# plt.savefig('%s.png'%matchWord, format='png',dpi=200)

# fig = plt.figure(figsize=(12, 8),dpi=1200)
# sns.lineplot(data=acc)
# plt.savefig('saving-a-high-resolution-seaborn-plot.png', dpi=1200)

# import plotly.graph_objects as go
# fig = go.Figure(data=acc)

# fig.update_layout(title='plotly.graph_objects', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))

# fig.show()
fig.write_html("testfile.html")
