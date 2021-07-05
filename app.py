# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 13:42:08 2021

@author: eden
"""

from TcyclesAnalyzerA01 import TcyclesAnalyzer
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

app=Flask(__name__)

app.secret_key = "secret key" # for encrypting the session
#It will allow below 16MB contents only, you can change it
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            devNameInd = filename.find('L1')
            deviceName  = filename[devNameInd:devNameInd+14]
            DevicePath = os.path.join(app.config['UPLOAD_FOLDER'], deviceName)
            if not os.path.isdir(DevicePath):
                os.mkdir(DevicePath) 
            filepath = os.path.join(DevicePath, filename)
            file.save(filepath)
            flash('File successfully Analyzed')
            print('\nfile name:'+filename)
            print('\nfile path:'+filepath)
            print('\nDevicePath:%s'%DevicePath)
            plot_folder = TcyclesAnalyzer(filename,DevicePath)
            print('\nFile successfully Analyzed in: %s'%plot_folder)
            return redirect('/')
        else:
            flash('Analysis has failed...')
            return redirect(request.url)


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000)