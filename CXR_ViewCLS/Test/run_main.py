"""
This model is classification in View position(PA or AP or Lateral or Others) of Chest X-ray. 
QI/Cardiac Team, Tae-Won Kim, Gyu-Jun Jeong
Asan hospital center, Medical Artifical Inteligence Researcher
Email : tlsgil2012@gmail.com
enviroment settings, Keras 2.4.3 Tensorflow 2.0.0(adding Pandas, skimage, PIL, Numpy ...)  
"""

import sys
sys.path.append('../')

import os
import argparse
import tqdm

import numpy as np
import tensorflow as tf
from src.utils import *
from model.model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View position of Chest X-ray classify')
    parser.add_argument('-i', help='Directory Path in Dicom data')
    parser.add_argument('-gpu', help='Gpu fan', default=0)
    parser.add_argument('--data_f', help='Input data file format', default='.dcm')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(str(args.gpu))

    ''' result record '''
    pred_results = record_result()    
    print(args.i)
    ''' dataset define '''
    for root, dirs, files in os.walk(args.i):
        if len(dirs) == 0:
            for i in range(len(files)):
                if str(args.data_f) in files[i]:
                    pred_results['file'].append(files[i])
                    pred_results['path'].append(os.path.join(root, files[i]))
                    pred_results['pred'].append('-')
                    
    print('a number of images N : {}'.format(len(pred_results['file'])))
                
    ''' model upload '''
    model = DenseNet169(input_size=(512, 512, 3), classes=4)
    compile_model(model)
    model.load_weights('../model/model_weight/11_0.037618.h5')

    ''' model inference '''
    label_info = {'0':'Lateral', '1':'PA', '2':'AP', '3':'Others','4':'Error'}
    
    for i in tqdm.tqdm(range(len(pred_results['path']))):
        cur_path = pred_results['path'][i]
        try:
            image = load_image(str(cur_path))
            pred_label = model.predict(image)[0]
            pred_index = str(np.argmax(pred_label))
            pred_results['pred'][i] = label_info[pred_index]
        except:
            pred_results['pred'][i] = label_info['4']
            
    write_result(pred_results, os.path.join(args.i, 'result.csv'))