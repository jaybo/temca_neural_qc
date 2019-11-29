#!/usr/bin/env python3
"""
Scores TEMCA QC images
"""


import argparse
import glob
import io
import json
import os
import sys
from pathlib import Path
from datetime import datetime  # for filename conventions
from time import time

# force to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py  # for saving the model
import numpy as np
import tensorflow as tf
#from google.colab import files
#from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from tensorflow import keras
from tensorflow.keras import layers, Model

#import altair as alt
import cv2
#import pandas as pd

print(tf.__version__)


def print_statistics(model, X_test, y_test):
    ''' predict probabilities for test set '''
    yhat_probs = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    print('Confusion Matrix :')
    print(confusion_matrix(y_test, yhat_classes))

    print('Accuracy Score :', accuracy_score(y_test, yhat_classes))
    print('Report : ')
    print(classification_report(y_test, yhat_classes))

all_types = [
    'quality_col_m',
    'quality_row_m',
    'focus_col_m',
    'distance_row_m',
    'distance_col_m',
    'std_dev_col_m',
]

def score_montage(model, apath):
    ''' given the CNN and a path, find and rescale the QC images, and run them through the model '''
    # todo, handle multiple ROIs
    full_path = apath / "0"

    factor = 6
    output_size = (int(635/factor), int((531-100)/factor))
    for entry in full_path.glob('*focus_col_m*'):
        images = []
        for index, t in enumerate(all_types):
            # concatenate all of the QC images together
            fin = str(entry).replace('focus_col_m', t)
            im = cv2.imread(fin)
            imCropped = im[:-100, :, :] # crop off the bottom
            imOut = cv2.resize (imCropped, output_size, 0, 0, cv2.INTER_AREA)
            imOut = imOut.astype(np.float32)
            if index == 0:
                images = imOut
            else:
                images = np.dstack((images, imOut))


        images = np.array([images,])
        images = (images / 255) - 0.5
        predictions = model.predict(images)
        return predictions

def main(args):
    ''' main entry point '''
    model = tf.keras.models.load_model('checkpoint.h5') 
    
    root_dir = args.directory[0]
    start = int(args.start[0])
    if args.end != -1:
        end = args.end
    else:
        end = start + 1

    # as six digit strings
    start_s = str(start).zfill(6)
    end_s = str(end).zfill(6)

    apath = Path(root_dir)
    all_dirs = list(apath.glob('**/'))
    filtered = []
    for d in all_dirs:
        # now filter only the desired directories
        if d.name >= start_s and d.name <= end_s:
            filtered.append(d)

    # hmm... sort filtered?
    true_pos = true_neg = false_pos = false_neg = 0
    for apath in filtered:
        predictions = score_montage(model, apath)
        if predictions is not None:
            if predictions[[0]] > 0.5:
                bad = 'BAD'
                if 'DONOTUSE' in str(apath):
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                bad = ''
                if 'DONOTUSE' in str(apath):
                    false_neg += 1
                else:
                    true_neg += 1
            print ("{0:s}, {1:.2f}, {2:s}".format(str(apath), float(predictions[[0]]), bad))
    
    total = true_pos + true_neg + false_pos + false_neg
    if total > 0:
        mess = f'''
            TP: {true_pos}, FP: {false_pos}
            FN: {false_neg}, TN: {true_neg}
            accuracy: {(true_pos + true_neg)/ (total)}
            '''
        print (mess)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("directory", nargs=1, help='root directory')
    parser.add_argument("start", nargs=1, help="starting aperture")
    parser.add_argument("end", nargs='?', default=-1, help="optional ending aperture")
    
    args = parser.parse_args()
    main(args)
