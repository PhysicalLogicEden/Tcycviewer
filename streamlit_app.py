# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:07:57 2021

@author: eden
"""
#pip install streamlit

import streamlit as st

   
# import PL own function from path
# from TcyclesAnalyzerA01 import TcyclesAnalyzer
import os


Analyzed_folder_path = 'W:\\Operation\\Production\\Production_tools_environment\\PTE\\uploads'

st.title("Physical-Logic Temp cycle Analyzer")

st.write('please choose Tcyc.txt file')

def save_uploadedfile(uploadedfile):
    with open(os.path.join("uploads",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to uploads".format(uploadedfile.name))
 
    
uploaded_file = st.file_uploader("Upload File",type=['txt']) 
if uploaded_file is not None:
   if uploaded_file.type == "text/plain":
      file_details = {"Filename":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
      devNameInd = uploaded_file.name.find('L1')
      deviceName  = uploaded_file.name[devNameInd:devNameInd+14]
      DevicePath = os.path.join(Analyzed_folder_path, deviceName)
      if not os.path.isdir(DevicePath):
              os.mkdir(DevicePath) 
              save_uploadedfile(uploaded_file)
      # plot_folder =  TcyclesAnalyzer(uploaded_file.name,uploaded_file)