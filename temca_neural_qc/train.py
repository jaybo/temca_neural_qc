#!/usr/bin/env python3
"""
Trains TEMCA neural QC network
Jay Borseth 2019.12.15
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import sys
import os

# force to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import h5py  # for saving the model
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from time import time
from datetime import datetime  # for filename conventions
import glob
import json
import argparse
# from PIL import Image
# from matplotlib import pyplot as plt
# # import altair as alt
# import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

# only thing we need to install is imgaug
#!pip install -q -U imgaug
#import imgaug
print(tf.__version__)


class TemcaNeuralTrainer:

    def __init__(self):
        self.data = []
        self.file_list_name = './file_list.json'
        self.all_name = './all.hd5'

    def write_hdf5_record(self, setname, metadata, data, hdf5_f, do_not_use=False):
        ''' each aperture becomes two datasets, data and metadata. 
        Also, a index dataset contains an attribute where the key is the setname 
        and value is True if do_not_use.
        Returns the key used to store the record '''

        metadata["do_not_use"] = do_not_use

        # add data and metadata
        try:
            hdf5_f.create_dataset(setname, data.shape, dtype=np.float32, data=data)
            hdf5_f.create_dataset(setname + '_meta', data=json.dumps(metadata))
        except OSError as ex:
            # already exists, recreate it
            print(ex, setname)
            del hdf5_f[setname]
            del hdf5_f[setname + '_meta']
            hdf5_f.create_dataset(setname, data.shape, dtype=np.float32, data=data)
            hdf5_f.create_dataset(setname + '_meta', data=json.dumps(metadata))

        # add to the index
        try:
            index = hdf5_f['index']
        except KeyError as ex:
            index = hdf5_f.create_dataset('index', dtype=np.bool)
        try:
            index.attrs[setname] = do_not_use
        except Exception as ex:
            print(ex)


    def read_hdf5_record(self, setname, hdf5_f):
        ''' Given a dataset name, return (meta, data, do_not_use)'''
        try:
            data = hdf5_f[setname][:,:,:]
            meta = json.loads(hdf5_f[setname + '_meta'][()])
            do_not_use = hdf5_f['index'].attrs[setname]
            return meta, data, do_not_use
        except KeyError as ex:
            # already exists, recreate it
            print (ex, setname)


    def read_hdf5_index(self, hdf5_f):
        ''' Returns a {} where the key is each dataset name, 
        and the value is do_not_use.
        '''
        index = {}

        try:
            for k in hdf5_f['index'].attrs.keys():
                v = hdf5_f['index'].attrs[k]
                #print (k, v)
                index[k] = v
            return index
        except KeyError as ex:
            # already exists, recreate it
            print (ex)


    def parse_metadata(self, meta_file):
        ''' Read in the metadata file into numpy arrays.
        returns a tuple (metadata, [Y,X,dataplanes])'''
        try:
            with open(meta_file) as data_file:
                json_data = json.load(data_file)
        except:
            raise Exception("Cannot find or parse metafile in: " + meta_file)

        metadata = json_data[0]['metadata']
        data = json_data[1]['data']
        max_rows = max_cols = 0

        # get max of rows and cols
        for tile in data:
            col, row = tile['img_meta']['raster_pos']
            max_rows = max(max_rows, row)
            max_cols = max(max_cols, col)

        # allocate identicaly sized data arrays
        mean = np.zeros((max_rows+1, max_cols+1), dtype=np.float32)
        mask = np.zeros((max_rows+1, max_cols+1), dtype=np.float32)
        focus = np.zeros((max_rows+1, max_cols+1), dtype=np.float32)
        std_dev = np.zeros((max_rows+1, max_cols+1), dtype=np.float32)

        # 6 planes, top_distance, top_x_offset, top_y_offset, side_distance, side_x_offset, side_y_offset,
        im_dist_to_ideal = np.zeros((max_rows+1, max_cols+1, 6), dtype=np.float32)
        # plane[0] = top, plane[1] = side
        im_match_quality = np.empty((max_rows+1, max_cols+1, 2), dtype=np.float32)
        im_match_quality.fill(-1)

        # fill in all the arrays
        for tile in data:
            col, row = tile['img_meta']['raster_pos']
            mask[row][col] = 1
            mean[row][col] = tile["mean"] / 256
            focus[row][col] = tile["focus_score"]
            # early metadata didn't have stddev
            if "std_dev" in tile:
                std_dev[row][col] =  tile["std_dev"]
            if 'matcher' in tile:
                mt = tile['matcher'][0]  # matcher top
                ms = tile['matcher'][1]  # matcher side
                # 2 left, 3 top, 4 right
                if mt['position'] != 3:
                    # swap them
                    ms, mt = (mt, ms)
                im_dist_to_ideal[row][col][0] = mt['distance']
                im_dist_to_ideal[row][col][1] = mt['dX']
                im_dist_to_ideal[row][col][2] = mt['dY']
                im_dist_to_ideal[row][col][3] = ms['distance']
                im_dist_to_ideal[row][col][4] = ms['dX']
                im_dist_to_ideal[row][col][5] = ms['dY']

                im_match_quality[row][col][0] = mt['match_quality']
                im_match_quality[row][col][1] = ms['match_quality']

        # finally, stack all planes together
        # planes = 12:    1      1     1       1          6                  2
        all_planes = np.dstack((mask, mean, focus, std_dev, im_dist_to_ideal, im_match_quality))
        #print (all_planes.shape)
        return (metadata, all_planes)


    def create_filelist(self, rootdir, start, end):
        """ Create list of files"""
        start = int(start)
        end = int(end)
        search_path = os.path.join(rootdir, r"**")
        search_path = os.path.join(search_path, r"_metadata*.json")
        files = glob.glob(search_path, recursive=True)
        files = sorted(files)
        print(len(files))

        # remove reference montages
        files = [f for f in files if not "_reference" in f]

        # only inclue apertures in the range start to end
        filtered = []
        for file in files:
            left, right = os.path.split(file) # right is metadata file
            left, right = os.path.split(left) # right is ROI index
            left, right = os.path.split(left) # right is barcode
            barcode = int(right)
            if barcode >= start and barcode <= end:
                filtered.append(file)
            pass

        print(len(filtered))

        # save a copy of the file names
        files_j = json.dumps(filtered, indent=2)
        with open(self.file_list_name, 'w') as f:
            f.write(files_j)
        return filtered


    def reload_filelist (self):
        """ Reload existing list of files """
        all_files = []
        with open(self.file_list_name, 'r') as f:
            all_files = json.load(f)
        all_files = sorted(all_files)
        return all_files


    def get_success_vs_failures(self, all_files):
        ''' segment the files based on DONOTUSE '''
        failures = [f for f in all_files if "DONOTUSE" in f]
        OKs = [f for f in all_files if "DONOTUSE" not in f]
        print (len(all_files), len(failures), "/", len(OKs))
        return all_files, failures, OKs


    def training_split(self, hdf5_file, split=0.25):         
        index = self.read_hdf5_index(hdf5_file)
        setnames = list(index.keys())
        Y = list(index.values())
        X = []
        with h5py.File(hdf5_file, 'r') as f:
            meta, data, do_not_use = self.read_hdf5_record(setnames[0], hdf5_file)
            pass

        # X.append(data)      #[None, ...]
        Y = []
        for k in range(64):
            X.append(data)
            Y.append(False)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.bool)
        print(X.shape)
        return X, Y


    def plot_history(self, histories, keys=['binary_crossentropy', 'accuracy']):
        plt.figure(figsize=(12, 8))

        for name, history in histories:
            #print(history.history.keys())
            for key in keys:
                val = plt.plot(history.epoch, history.history['val_'+key],
                                '--', label=name.title()+' Val')
                plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                        label=name.title()+' Train')

        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()

        plt.xlim([0, max(history.epoch)])

    # """## Scatterplot of sizes and frequency of occurance"""

    # df = pd.DataFrame(name_and_size_d)

    # alt.Chart(df).mark_circle(size=60).encode(
    #     x=alt.X('w', bin = False, scale=alt.Scale(zero=False)),
    #     y=alt.Y('h', bin = False, scale=alt.Scale(zero=False)),
    #     size='count()',
    #     tooltip=['w', 'h']
    # ).interactive()





    # def model_statistics(self, x, y):
    #     # predict probabilities for test set
    #     global yhat_probs
    #     global yhat_classes
    #     yhat_probs = model.predict(x, verbose=0)
    #     # predict crisp classes for test set
    #     yhat_classes = model.predict_classes(x, verbose=0)
    #     # reduce to 1d array
    #     yhat_probs = yhat_probs[:, 0]
    #     yhat_classes = yhat_classes[:, 0]

    #     print('Confusion Matrix :')
    #     cm = confusion_matrix(y, yhat_classes)
    #     print(cm)
    #     print('Accuracy Score :', accuracy_score(y, yhat_classes))
    #     print('Report : ')
    #     print(classification_report(y, yhat_classes))

    #model_statistics(X_test, y_test)
    #model_statistics(X_train, y_train)

    def get_model(self):
        loss = 0.0001
        drop = 0.4
        spatial_drop = 0.1

        model = keras.models.Sequential([
            keras.layers.Input(shape=(None, None, 12)),
            
            #keras.layers.Dropout(drop),

            #keras.layers.BatchNormalization(),

            keras.layers.Convolution2D(filters=32, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=32, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),

            keras.layers.Convolution2D(filters=64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),

            keras.layers.Convolution2D(filters=128, kernel_size = (1,1)),

            keras.layers.GlobalMaxPooling2D(),
            
            keras.layers.Dense(1, activation='sigmoid')

            #keras.layers.BatchNormalization(),
            # keras.layers.MaxPooling2D(),
            # #keras.layers.SpatialDropout2D(spatial_drop),
            # keras.layers.Dropout(drop),
            
            # keras.layers.Convolution2D(filters=64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # keras.layers.Convolution2D(filters=64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # #keras.layers.BatchNormalization(),
            # keras.layers.MaxPooling2D(),
            # #keras.layers.SpatialDropout2D(spatial_drop),
            # keras.layers.Dropout(drop),

            # #keras.layers.Convolution2D(filters=128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # #keras.layers.Convolution2D(filters=128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # #keras.layers.Convolution2D(filters=128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # #keras.layers.BatchNormalization(),
            # #keras.layers.MaxPooling2D(),
            # #keras.layers.Dropout(drop),

            # #keras.layers.Convolution2D(filters=256, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # #keras.layers.Convolution2D(filters=256, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # #keras.layers.Convolution2D(filters=256, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # #keras.layers.BatchNormalization(),
            # keras.layers.MaxPooling2D(),
            # keras.layers.Dropout(drop),

            # keras.layers.Convolution2D(filters=64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # #keras.layers.Convolution2D(filters=64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # #keras.layers.Convolution2D(filters=128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            # keras.layers.MaxPooling2D(),
            # keras.layers.Dropout(drop),
            
            # #keras.layers.GlobalAveragePooling2D(),


            # #keras.layers.Flatten(),
            # #keras.layers.Dropout(drop),


            # #keras.layers.Dense(4096, activation='relu'),
            # #keras.layers.Dense(4096, activation='relu'),

            # # keras.layers.Dense(1, activation='sigmoid')

            # #keras.layers.Dense(4096, activation='relu'),
            # #keras.layers.Dense(4096, activation='relu'),
            # #keras.layers.Dense(1, activation='sigmoid'),

        ])

        #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        opt = keras.optimizers.Adam()
        #opt = keras.optimizers.SGD(lr=0.001, decay=.01, momentum=0.9, nesterov=True)
        model.compile(opt, loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
        return model


    def train(self, model, X_train, y_train, X_test, y_test):
        


        logdir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(logdir)
        baseline_history = model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test),
                callbacks=[
                    keras.callbacks.ModelCheckpoint('./checkpoint.h5',
                                monitor='loss', verbose=0, save_best_only=True, save_weights_only=False),
                    EarlyStopping(patience=15, restore_best_weights=True,
                                monitor='loss'),
                    ReduceLROnPlateau(patience=6, factor=0.7,
                                min_lr=0.000001, verbose=1),
                    keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        ])

        # model.evaluate(X_test, y_test, verbose=2)
        # plot_history([('baseline', baseline_history)])
        # model_statistics(X_test, y_test)

        # no_match = np.logical_xor (yhat_classes.astype('bool'), y_test.astype('bool'))
        # no_match_idx = np.nonzero(no_match)
        # no_match_idx = list(no_match_idx[0])
        # #no_match_index
        # #name_and_size_d[list(no_match_idx[0].ravel())]
        # [name_and_size_d[j] for j in no_match_idx]

        # y_test[no_match_idx]

        # print (len(X))
        # failures = 0
        # for j in range(len(X)):
        #   k = np.array([X[j]],)
        #   result = model.predict(k) > 0.5
        #   if result != Y[j]:
        #     print (j, Y[j], result)
        #     failures += 1
        # print (f'failures: {failures}')

        # X[0,0,0,:]

        # foo = X_test[no_match_idx]
        # foo2 = y_test[no_match_idx]
        # print (foo.shape)
        # print (foo2.shape)
        # print (foo2)
        # model.predict ([X_test[no_match_idx]]) > 0.5

        # """##Save checkpoint"""

        # model.save_weights('/content/gdrive/My Drive/model_qc/model_checkpoint.h5')

        # model.load_weights('/content/gdrive/My Drive/model_qc/model_checkpoint.h5')

        # model = keras.models.load_model('/content/gdrive/My Drive/model_qc/checkpoint.h5')



        # model.save('/content/gdrive/My Drive/model_qc/checkpoint.h5')

    # def kfold(self):
    #     """## KFold"""

    #     # Evaluate model using standardized dataset. 
    #     estimators = []
    #     #estimators.append(('standardize', StandardScaler()))
    #     estimators.append(('mlp', keras.wrappers.scikit_learn.KerasClassifier(build_fn=get_model, epochs=32, batch_size=32, verbose=0)))
    #     pipeline = Pipeline(estimators)
    #     kfold = StratifiedKFold(n_splits=5, shuffle=True)
    #     results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
    #     print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    #     model.summary()

def main():
    ''' main entry point '''

    parser = argparse.ArgumentParser()

    parser.add_argument("directory", nargs=1, help='root directory', default="C:/Users/jaybo/Google Drive/data/jay7/000000/0/")
    parser.add_argument("start", nargs='?', help="optional starting aperture", default="000000")
    parser.add_argument("end", nargs='?', help="optional ending aperture", default="999999")
    
    args = parser.parse_args()

    tnt = TemcaNeuralTrainer()

    for directory in args.directory:
        files = tnt.create_filelist(directory, args.start, args.end)

    index = {}
    files = tnt.reload_filelist()

    with h5py.File(tnt.all_name, 'a') as hdf5_f:
        for i, meta_file_path in enumerate(files):
            setname = os.path.split(meta_file_path)[1].replace('_metadata_', '').replace('.json', '')
            do_not_use="DONOTUSE" in meta_file_path
            if do_not_use:
                print(i, setname, "DONOTUSE")
            else:
                print(i, setname)
            try:
                meta, data = tnt.parse_metadata(meta_file_path)
                tnt.write_hdf5_record(setname, meta, data, hdf5_f, do_not_use=do_not_use)
                # meta, data, do_not_use = tnt.read_hdf5_record(setname, hdf5_f)
                # index = tnt.read_hdf5_index(hdf5_f)
            except Exception as ex:
                print (ex)

        # for j in range(1000):
        #     tnt.write_hdf5_record(setname, meta, data, hdf5_f)
        #     meta, data = tnt.read_hdf5_record(setname, hdf5_f)


    # X, Y = tnt.training_split(tnt.all_name)

    # model = tnt.get_model()
    # model.summary()

    # tnt.train(model, X, Y, X, Y)

    pass

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
