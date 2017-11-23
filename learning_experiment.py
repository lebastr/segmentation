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

class BatchGenerator(object):
    def __init__(self, data_set, ids, shuffle=True, shuffle_on_each_epoch=False, random_rotate=False,
                 random_translate=False, crop_mask=0):

        self.ids = copy.copy(ids)
        self.data_set = data_set
        self.random_rotate = random_rotate
        self.random_translate = random_translate
        self.shuffle_on_each_epoch = shuffle_on_each_epoch
        self.crop_mask = crop_mask

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

            def transformation(crop = 0):
                def g(a,b):
                    return lambda x: b(a, x)

                transform = lambda i: i
                if self.random_rotate or self.random_translate:
                    transform = g(transform, lambda t,i: t(i).rotate(angle, translate=(xt, yt)))

                if crop > 0:
                    transform = g(transform, lambda t,i: t(i).crop((crop,crop, i.height-crop, i.width-crop)))

                return transform

            return transformation

        if self.shuffle_on_each_epoch:
            random.shuffle(self.ids)
            
        for i in range(0, len(self.ids), batch_size):
            ids = self.ids[i:i+batch_size]
            Xs = []
            Ys = []
            for img_id in ids:
                transformation = make_transformation()
                Xs.append(dataset.apply_pil_transform(transformation(), self.data_set.get_ndarray(img_id)))
                Ys.append(dataset.apply_pil_transform(transformation(crop = self.crop_mask),
                                                      (self.data_set.get_mask(img_id)[:,:,None])))

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
