import os
import re
import json
import unet
import dataset
import train_test_loader
import datetime
import copy
import random

from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt

import PIL.Image

def create_experiment(description, data_set):
    DATE = str(datetime.datetime.now())
    DIR = './experiments/%s' % DATE

    os.mkdir(DIR)

    json.dump(description, open(DIR + "/description.txt", "w"))

    with open(DIR + "/data_set.json", 'w') as f:
        f.write(data_set.dumps())

    for m in ["dataset", "unet", "train_test_loader", "learning_experiment"]:
        fname = "%s.py" % m
        copyfile(fname, DIR + '/' + fname)

    print "Create experiment in directory: %s" % DIR
    return LearningExperiment(DIR)

class LearningExperiment(object):
    def __init__(self, path):
        self.path = path
        self.description = json.load(open(self.path + "/description.txt", 'r'))
        self.__model_dir__ = self.path + "/model"
        self.__model_history_path__ = self.__model_dir__ + "/history.json"
        self.__model_description_path__ = self.__model_dir__ + "/description.json"

        data_set_path = self.path + "/data_set.json"
        if os.path.exists(data_set_path):
            self.data_set = dataset.load_data_set_from_dumps(open(data_set_path, 'r').read())
        else:
            self.data_set = None
        
    def epochs(self):
        return sorted([int(x) for x in os.listdir(self.path + '/model') if re.match(r'^\d+$', x)])
    
    def load_net(self, epoch = None, input_shape = None):
        if os.path.exists(self.__model_description_path__):
            net_description = json.load(open(self.__model_description_path__, 'r'))
        else:
            net_description = None

        if net_description is not None:
            name = net_description['name']
            if name == 'unet':
                net = unet.UnetModel(net_description['input_shape'])

            elif name == 'vgg-unet':
                net = unet.VGGUnetModel()

            elif name == 'vgg-unet-with-crop':
                net = unet.VGGUnetModelWithCrop(net_description['N'])

            else:
                raise "Unknown net name!"

        else: # old UnetModel experiment
            if input_shape is None:
                assert self.data_set is not None, "Pass explicit input_shape!"
                input_shape = self.data_set.shape

            net = unet.UnetModel(input_shape)

        if epoch is None:
            epoch = self.epochs()[-1]
            
        net.load_weights(self.__model_dir__ + '/%d' % epoch)
        net.u_history = self.loss_history()
        
        return net
    
    def loss_history(self):
        return json.load(open(self.__model_history_path__, 'r'))
        
    def save(self, net, fname):
        if not os.path.exists(self.__model_dir__):
            os.mkdir(self.__model_dir__)

        json.dump(net.net_description, open(self.__model_description_path__, 'w'))
        json.dump(net.u_history, open(self.__model_history_path__, "w"))
        net.save(self.__model_dir__ + "/" + fname)

    def train(self, net, batch_generator, epochs=10, batch_size=10):
        for _ in range(epochs):
            epoch = len(net.u_history)
            losses = []
            net.u_history.append({'epoch': epoch, 'loss_array' : losses})

            try:
                for ids, X, Y in batch_generator(batch_size):
                    loss = net.train_on_batch(X, Y)
                    losses.append(float(loss))
                    print("epoch: %d, loss: %f" % (epoch, loss))
            except BaseException as e:
                self.save(net, "epoch_%d_interrupted" % epoch)
                raise e

            self.save(net, "%d" % epoch)


def expand_with_mirrors(X, padding):
    assert X.shape[0] == X.shape[1], "X must have square form"
    size = X.shape[0]

    new_size = size + 2 * padding
    X_expanded = np.zeros((size + 2 * padding, size + 2 * padding, X.shape[2]), dtype=X.dtype)

    X_expanded[padding:size + padding, padding:size + padding,:] = X[:,:,:]

    X_v = X[::-1,:,:]
    X_h = X[:, ::-1,:]
    X_hv = X_h[::-1,:,:]

    X_expanded[:padding, padding:new_size - padding,:] = X_v[size - padding:,:,:]
    X_expanded[new_size - padding:, padding:new_size - padding,:] = X_v[:padding, :,:]
    X_expanded[padding:new_size - padding, :padding,:] = X_h[:, size - padding:,:]
    X_expanded[padding:new_size - padding, new_size - padding:,:] = X_h[:, :padding,:]
    X_expanded[:padding, :padding,:] = X_hv[size - padding:, size - padding:,:]
    X_expanded[:padding, new_size - padding:,:] = X_hv[size - padding:, :padding,:]
    X_expanded[new_size - padding:, new_size - padding:,:] = X_hv[:padding, :padding,:]
    X_expanded[new_size - padding:, :padding,:] = X_hv[:padding, size - padding:,:]

    return X_expanded


def symmetry_crop(X, s):
    h, w = X.shape[:2]
    return X[s:h-s, s:w-s, :]

class BatchGenerator(object):
    def __init__(self, data_set, ids, shuffle=True, shuffle_on_each_epoch=False,
                 input_size=None, crop=0, random_rotation_amplitude=0, channel_first=False):

        self.ids = copy.copy(ids)
        self.data_set = data_set
        self.shuffle_on_each_epoch = shuffle_on_each_epoch
        self.channel_first = channel_first

        if shuffle:
            random.shuffle(self.ids)

        assert data_set.shape[0] == data_set.shape[1], "Image must be a square!"

        if input_size is None:
            input_size = data_set.shape[0]

        self.image_size = data_set.shape[0]
        self.input_size = input_size
        self.random_rotation_amplitude = random_rotation_amplitude
        self.crop = crop

    def __call__(self, batch_size):
        def make_random_rotate():
            alpha = np.random.uniform(-self.random_rotation_amplitude, self.random_rotation_amplitude)
            return lambda img: PIL.Image.Image.rotate(img, alpha)

        def make_random_crop():
            ext_size = self.image_size + 2*self.crop
            low = 0
            high = ext_size - self.input_size
            [x_corner,y_corner] = np.int32(np.random.uniform(low, high, size=2))

            def crop(X):
                return X[x_corner:x_corner+self.input_size, y_corner:y_corner + self.input_size, :]

            return crop

        if self.shuffle_on_each_epoch:
            random.shuffle(self.ids)
            
        for i in range(0, len(self.ids), batch_size):
            ids = self.ids[i:i+batch_size]
            Xs = []
            Ys = []

            for img_id in ids:
                X = self.data_set.get_ndarray(img_id)
                Y = self.data_set.get_mask(img_id)[:,:,None]

                if self.crop > 0:
                    X = expand_with_mirrors(X, self.crop)
                    Y = expand_with_mirrors(Y, self.crop)

                crop = make_random_crop()
                rotate = make_random_rotate()

                X = crop(X)
                X = dataset.apply_pil_transform(rotate, X)

                Y = crop(Y)
                Y = symmetry_crop(Y, self.crop)
                Y = dataset.apply_pil_transform(rotate, Y)

                Xs.append(X)
                Ys.append(Y)

            Xs = np.array(Xs)
            Ys = np.array(Ys)

            if self.channel_first:
                Xs = Xs.transpose([0,3,1,2])
                Ys = Ys.transpose([0,3,1,2])

            yield ids, Xs, Ys

def moving_average(window_size, xs):
    assert window_size % 2 == 1, "windows_size must be odd"
    arr = np.empty(xs.shape[0] + 1 - window_size)
    arr[0] = xs[:window_size].mean()
    for j in range(0, arr.shape[0] - 1):
        arr[j+1] = arr[j] + 1.0/window_size * (xs[j+window_size] - xs[j])
    return arr

def plot_loss(history, window_size=1, ax = None, color='b'):
    epoch_x = [0]
    loss = []
    for v in history:
        epoch_x.append(len(v['loss_array']) + epoch_x[-1])
        loss = loss + v['loss_array']

    if ax is None:
        prefix = plt
    else:
        prefix = ax
        
    for x in epoch_x:
        prefix.axvline(x, alpha=0.2)
    
    loss = np.array(loss)
    prefix.plot(moving_average(window_size, loss), ',', color=color)
