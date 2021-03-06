#!/usr/bin/env python3
"""
Trains TEMCA neural QC network
Jay Borseth 2019.12.15
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import argparse
import glob
import io
import json
import os
import pickle
import sys
from datetime import datetime
from time import time

import cv2
import h5py 
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# force to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(tf.__version__)


class TemcaNeuralTrainer:

    def __init__(self):
        self.data = []
        self.file_list_name = './file_list.json'
        self.all_name = './all.hd5'
        self.scaler_file = './scaler.bin'
        self.scalers = {}

        # which planes to use from the full dataset?
        # planes = 12:             1      1     1       1          6                  2
        # all_planes = np.dstack((mask, mean, focus, std_dev, im_dist_to_ideal, im_match_quality))
        self.include_planes = [0, 1, 2, 3, 10]

        self.max_shape = (48, 64, len(self.include_planes))  # padded size
        self.ignore = ['reference', 'index', 'test', 'meta', 'back']

    ##------------------------------ HDF5 ---------------------------------------##

    def write_hdf5_record(self, setname, metadata, data, hdf5_f, do_not_use=False):
        ''' each aperture becomes two datasets, data and metadata. 
        Also, a index dataset contains an attribute where the key is the setname 
        and value is True if do_not_use.
        Returns the key used to store the record '''

        metadata["do_not_use"] = do_not_use

        # add data and metadata
        try:
            hdf5_f.create_dataset(setname, data.shape, data=data)
            hdf5_f.create_dataset(setname + '_meta', data=json.dumps(metadata))
        except OSError as ex:
            # bad key
            print(ex, setname)
            del hdf5_f[setname]
            del hdf5_f[setname + '_meta']
            hdf5_f.create_dataset(setname, data.shape, data=data)
            hdf5_f.create_dataset(setname + '_meta', data=json.dumps(metadata))

        # add to the index
        try:
            index = hdf5_f['index']
        except KeyError as ex:
            index = hdf5_f.create_dataset('index', data="Maybe it needs data to set attributes?")
        try:
            myattr = index.attrs.get(setname)
            if myattr is not None:
                index.attrs.modify(setname, np.bool(do_not_use))
            else:
                index.attrs[setname] = np.bool(do_not_use)
        except Exception as ex:
            print('modifying or creating attr: ', ex)


    def read_hdf5_record(self, setname, hdf5_f):
        ''' Given a dataset name, return (meta, data, do_not_use)'''
        try:
            data = hdf5_f[setname][:,:,:]
            meta = json.loads(hdf5_f[setname + '_meta'][()])
            do_not_use = hdf5_f['index'].attrs.get(setname)
            return meta, data, do_not_use
        except KeyError as ex:
            # bad key
            print (ex, setname)

    def read_hdf5_data(self, setname, hdf5_f):
        ''' Given a dataset name, return (meta, data, do_not_use)'''
        try:
            data = hdf5_f[setname][:, :, :]
            return  data
        except KeyError as ex:
            # bad key
            print (ex, setname)

    def read_hdf5_index(self, hdf5_f):
        ''' Returns a {} where the key is each dataset name, 
        and the value is do_not_use.
        '''
        index = {}

        try:
            for k in hdf5_f['index'].attrs.keys():
                v = hdf5_f['index'].attrs.get(k)
                if 'index' in k or 'meta' in k:
                    continue
                #print (k, v)
                index[k] = v
            return index
        except KeyError as ex:
            # bad key
            print (ex)

    def check_hd5_consistency(self, hdf5_f):
        ''' verify that all index entries are present and not null'''
        print ('consistency check')
        index = self.read_hdf5_index(hdf5_f)
        keys = hdf5_f.keys()
        len_index = len(index)
        len_keys = len(keys)
        
        # (data + meta) * apertures + index
        print (f'len_index: {len_index}, len_keys: {len_keys}, keys and index legnth match?: {2 * len_index + 1}')
        
        # check for non null and count positives and negatives
        none_count = pos = neg = 0
        for k in keys:
            v = index.get(k)
            if v is None:
                none_count += 1
            elif v:
                pos += 1
            else:
                neg += 1
        print (f'pos: {pos}, neg: {neg}, None: {none_count}')

    def XY_from_hdf5(self, hdf5_f):
        ''' get X, Y arrays from hdf5 '''
        keys = hdf5_f.keys()
        X = []
        Y = []
        for k in keys:
            # print (k)
            if any(x in k.lower() for x in self.ignore):
                continue
            data = self.read_hdf5_data(k, hdf5_f)
            # zero padding
            padded = np.zeros(self.max_shape)
            padded[0:data.shape[0], 0:data.shape[1], :] = data[:, :, self.include_planes]
            X.append(padded)
            Y.append(hdf5_f['index'].attrs.get(k))
        return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.bool)

    ##--------------------------- Metadata Parser ------------------------------##

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
        metadata["rows_cols"] = (max_rows, max_cols)

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


    ##------------------------------ filelist ---------------------------------------##

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
        files = [f for f in files if not "_reference" in f or not "back_up" in f or not "test" in f]


        # only inclue apertures in the range start to end
        filtered = []
        for file in files:
            try:
                left, right = os.path.split(file) # right is metadata file
                left, right = os.path.split(left) # right is ROI index
                left, right = os.path.split(left) # right is barcode
                barcode = int(right)
                if barcode >= start and barcode <= end:
                    filtered.append(file)
            except Exception as ex:
                print (ex, file)

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

    ##----------------------- Training split ----------------------------##

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


    # def plot_history(self, histories, keys=['binary_crossentropy', 'accuracy']):
    #     plt.figure(figsize=(12, 8))

    #     for name, history in histories:
    #         #print(history.history.keys())
    #         for key in keys:
    #             val = plt.plot(history.epoch, history.history['val_'+key],
    #                             '--', label=name.title()+' Val')
    #             plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
    #                     label=name.title()+' Train')

    #     plt.xlabel('Epochs')
    #     plt.ylabel(key.replace('_', ' ').title())
    #     plt.legend()

    #     plt.xlim([0, max(history.epoch)])



    # """## Scatterplot of sizes and frequency of occurance"""

    # df = pd.DataFrame(name_and_size_d)

    # alt.Chart(df).mark_circle(size=60).encode(
    #     x=alt.X('w', bin = False, scale=alt.Scale(zero=False)),
    #     y=alt.Y('h', bin = False, scale=alt.Scale(zero=False)),
    #     size='count()',
    #     tooltip=['w', 'h']
    # ).interactive()





    def model_statistics(self, model, x, y):
        # predict probabilities
        yhat_probs = model.predict(x, verbose=0)
        # predict crisp classes for test set
        yhat_classes = model.predict_classes(x, verbose=0)
        # reduce to 1d array
        yhat_probs = yhat_probs[:, 0]
        yhat_classes = yhat_classes[:, 0]

        print('Confusion Matrix :')
        cm = confusion_matrix(y, yhat_classes)
        print(cm)
        print('Accuracy Score :', accuracy_score(y, yhat_classes))
        print('Report : ')
        print(classification_report(y, yhat_classes))
        return yhat_probs, yhat_classes

        #model_statistics(X_test, y_test)
        #model_statistics(X_train, y_train)

    ##------------------------------ Model ---------------------------------------##

    def get_model(self):
        loss = 0.0001
        drop = 0.1
        spatial_drop = 0.1

        model = keras.models.Sequential([
            keras.layers.Input(shape=self.max_shape),
            
            keras.layers.BatchNormalization(),
            
            keras.layers.Convolution2D(filters=64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            #keras.layers.SpatialDropout2D(spatial_drop),
            keras.layers.Dropout(drop),

            keras.layers.Convolution2D(filters=128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(drop),

            keras.layers.Convolution2D(filters=256, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=256, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=256, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(drop),

            keras.layers.Convolution2D(filters=512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(drop),
            

            keras.layers.Convolution2D(filters=512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.Convolution2D(filters=512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', kernel_regularizer=keras.regularizers.l2(loss)),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(drop),

            keras.layers.Flatten(),
            # keras.layers.Dropout(drop),

            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dense(4096, activation='relu'),

            keras.layers.Dense(1, activation='sigmoid')

        ])

        #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        opt = keras.optimizers.Adam()
        #opt = keras.optimizers.SGD(lr=0.001, decay=.01, momentum=0.9, nesterov=True)
        model.compile(opt, loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
        return model


    ##------------------------------ Train ---------------------------------------##

    def train(self, model, X_train, y_train, X_test, y_test):

        logdir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(logdir)
        
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
        class_weights = dict(enumerate(class_weights))

        baseline_history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights,
                callbacks=[
                    keras.callbacks.ModelCheckpoint('./checkpoint.h5',
                                monitor='loss', verbose=0, save_best_only=True, save_weights_only=False),
                    EarlyStopping(patience=15, restore_best_weights=True,
                                monitor='loss'),
                    ReduceLROnPlateau(patience=6, factor=0.7,
                                min_lr=0.000001, verbose=1),
                    keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        ])

        model.evaluate(X_test, y_test, verbose=2)
        #plot_history([('baseline', baseline_history)])
        yhat_prob, yhat_classes = self.model_statistics(model, X_test, y_test)

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

    parser = argparse.ArgumentParser(description='TEMCA Neural QC')

    subparsers = parser.add_subparsers(dest="subparser")

    # add more data to the dataset
    add_data = subparsers.add_parser(
        'add_data', help='add to datset')
    add_data.add_argument("-d", "--directory", type=str, help='root directory', default="C:/Users/jaybo/Google Drive/data/jay7/000000/0/")
    add_data.add_argument("-s", "--start", type=str, help="optional starting aperture", default="000000")
    add_data.add_argument("-e", "--end", type=str, help="optional ending aperture", default="999999")
    add_data.add_argument("-r", "--reload_filelist", type=bool, help="reload instead of recursive directory search", default=False)

    # normalize
    normalize = subparsers.add_parser(
        'normalize', help='calculate normalization')

    # train
    train = subparsers.add_parser(
        'train', help='train the network')

    # test
    test = subparsers.add_parser(
        'test', help='test')

    args = parser.parse_args()

    tnt = TemcaNeuralTrainer()

    ##------------------------------ add_data ---------------------------------------##
    if args.subparser == 'add_data':

        index = {}
        if args.reload_filelist:
            files = tnt.reload_filelist()
        else:
            files = tnt.create_filelist(args.directory, args.start, args.end)

        with h5py.File(tnt.all_name, 'a') as hdf5_f:
            for i, meta_file_path in enumerate(files):
                hdf5_f.flush()
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

    ##------------------------------ normalize ---------------------------------------##
    if args.subparser == 'normalize':

        index = {}
        tnt.scalers = {}

        with h5py.File(tnt.all_name, 'r') as hdf5_f:
            index = tnt.read_hdf5_index(hdf5_f)
            for setname in index.keys():
                print(setname)
                meta, data, do_not_use = tnt.read_hdf5_record(setname, hdf5_f)

                if not tnt.scalers: 
                    # create a scaler for each channel
                    for i in range(data.shape[-1]):
                        tnt.scalers[str(i)] = StandardScaler()

                for i in range(data.shape[-1]):
                    tnt.scalers[str(i)].partial_fit(data[:,:,i].transpose())

        for i in range(data.shape[-1]):
            pickle.dump(tnt.scalers[str(i)], open(tnt.scaler_file + str(i), 'wb'))


        # scaler = pickle.load(open(tnt.scaler_file, 'rb'))
        # test_scaled_set = scaler.transform(test_set)


    ##------------------------------ train ---------------------------------------##
    if args.subparser == 'train':
        with h5py.File(tnt.all_name, 'r') as hdf5_f:
            # tnt.check_hd5_consistency(hdf5_f)
            X, Y = tnt.XY_from_hdf5(hdf5_f)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)
        model = tnt.get_model()
        model.summary()

        tnt.train(model, X_train, Y_train, X_test, Y_test)

        pass

    ##------------------------------ test ---------------------------------------##
    if args.subparser == 'test':
        with h5py.File(tnt.all_name, 'r') as hdf5_f:
            # tnt.check_hd5_consistency(hdf5_f)
            X, Y = tnt.XY_from_hdf5(hdf5_f)

        pass

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
