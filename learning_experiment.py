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
        self.description = open(self.path + "/description.txt", 'r').read()
        
        data_set_path = self.path + "/data_set.json"
        if os.path.exists(data_set_path):
            self.data_set = dataset.load_data_set_from_dumps(open(data_set_path, 'r').read())
        else:
            self.data_set = None
        
    def epochs(self):
        return sorted([int(x) for x in os.listdir(self.path + '/model') if re.match(r'^\d+$', x)])
    
    def load_net(self, epoch = None, input_shape=None):
        if input_shape is None:
            assert self.data_set is not None, "Pass explicit input_shape!"
            input_shape = self.data_set.shape
            
        net = unet.UnetModel(input_shape)
        if epoch is None:
            epoch = self.epochs()[-1]
            
        net.load_weights(self.path + '/model/%d' % epoch)
        net.u_history = self.loss_history()
        
        return net
    
    def loss_history(self):
        return json.load(open(self.path + '/model/history.json', 'r'))
        
    def save(self, net, fname):
        persistent_dir = self.path + "/model"
        if not os.path.exists(persistent_dir):
            os.mkdir(persistent_dir)

        json.dump(net.u_history, open(persistent_dir + "/history.json", "w"))
        net.save(persistent_dir + "/" + fname)

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

class BatchGenerator(object):
    def __init__(self, data_set, ids, shuffle=True, shuffle_on_each_epoch=False, random_rotate=False, random_translate=False):
        self.ids = copy.copy(ids)
        self.data_set = data_set
        self.random_rotate = random_rotate
        self.random_translate = random_translate
        self.shuffle_on_each_epoch = shuffle_on_each_epoch
        
        (h,w,d) = data_set.get_ndarray(self.ids[0]).shape
        self.image_size = (h,w)
        self.n_channels = d
                
        if shuffle:
            random.shuffle(self.ids)
        
    def shape(self):
        return (self.image_size[0], self.image_size[1], self.n_channels)
    
    def __call__(self, batch_size):
        def make_transformation():
            if self.random_rotate:
                angle = np.random.uniform(0.0, 360.0)
            else: angle = 0
            
            if self.random_translate:
                xt = np.random.uniform(-50.0,50.0)
                yt = np.random.uniform(-50.0,50.0)
            else:
                xt = 0
                yt = 0
                
            def transformation(X):
                if self.random_rotate or self.random_translate:
                    return dataset.rotate_and_translate(X, angle, translate=(xt, yt))
                else:
                    return X
            return transformation


        if self.shuffle_on_each_epoch:
            random.shuffle(self.ids)
            
        for i in range(0, len(self.ids), batch_size):
            ids = self.ids[i:i+batch_size]
            Xs = []
            Ys = []
            for img_id in ids:
                transformation = make_transformation()
                Xs.append(transformation(self.data_set.get_ndarray(img_id)))
                Ys.append(transformation(self.data_set.get_mask(img_id)[:,:,None]))

            Xs = np.array(Xs)
            Ys = np.array(Ys)
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
