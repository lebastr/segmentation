import numpy as np
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, Reshape,
                          BatchNormalization, Concatenate)

from keras.optimizers import Adam, Nadam

import random
import copy
import os
import json

def make_conv_block(nb_filters, input_tensor, block):
    def make_stage(input_tensor, stage):
        name = 'conv_{}_{}'.format(block, stage)
        x = Conv2D(nb_filters, (3, 3), activation='relu',
                   padding='same', name=name)(input_tensor)
        name = 'batch_norm_{}_{}'.format(block, stage)
        x = BatchNormalization(name=name)(x)
        x = Activation('relu')(x)
        return x

    x = make_stage(input_tensor, 1)
    x = make_stage(x, 2)
    return x

class UnetModel(Model):
    def __init__(self, name, dataset, persistent_dir, lr = 1e-4):
        self.u_history = []
        self.dataset = dataset
        self.persistent_dir = persistent_dir
        
        input_shape = self.dataset.shape
        
        inputs = Input(input_shape)
        conv1 = make_conv_block(32, inputs, 1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = make_conv_block(64, pool1, 2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = make_conv_block(128, pool2, 3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = make_conv_block(256, pool3, 4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = make_conv_block(512, pool4, 5)

        up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = make_conv_block(256, up6, 6)

        up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = make_conv_block(128, up7, 7)

        up8 = Concatenate()([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = make_conv_block(64, up8, 8)

        up9 = Concatenate()([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = make_conv_block(32, up9, 9)

        conv10 = Conv2D(input_shape[2], (1, 1), activation='sigmoid', name='conv_10_1')(conv9)
    
    
        super(UnetModel, self).__init__(inputs=inputs, outputs=conv10, name=name)
        self.compile(optimizer=Nadam(lr=1e-4), loss='binary_crossentropy')

    def train(self, batch_generator, train_ids, batch_size=10, epochs=10):
        gen = batch_generator(self.dataset, train_ids)
        for _ in range(epochs):
            epoch = len(self.u_history)
            losses = []
            self.u_history.append({'epoch': epoch, 'loss_array' : losses})
            
            try:
                for ids, X, Y in gen(batch_size):
                    loss = self.train_on_batch(X, Y)
                    losses.append(float(loss))
                    print("epoch: %d, loss: %f" % (epoch, loss))
            except BaseException as e:
                self.save("epoch_%d_interrupted" % epoch)
                raise e
            
            self.save("%d" % epoch)
        
    def save(self, fname):
        if not os.path.exists(self.persistent_dir):
            os.mkdir(self.persistent_dir)

        json.dump(self.u_history, open(self.persistent_dir + "/history.json", "w"))
        super(UnetModel, self).save(self.persistent_dir + "/" + fname)

        
class BatchGenerator(object):
    def __init__(self, data_set, ids, shuffle=True):
        self.ids = copy.copy(ids)
        self.data_set = data_set
        
        (h,w,d) = data_set.get_ndarray(self.ids[0]).shape
        self.image_size = (h,w)
        self.n_channels = d
                
        if shuffle:
            random.shuffle(self.ids)
        
    def shape(self):
        return (self.image_size[0], self.image_size[1], self.n_channels)
    
    def __call__(self, batch_size):
        for i in range(0, len(self.ids), batch_size):
            ids = self.ids[i:i+batch_size]
            Xs = np.array([self.data_set.get_ndarray(img_id) for img_id in ids])
            Ys = np.array([self.data_set.get_mask(img_id)[:,:,None] for img_id in ids])
            yield ids, Xs, Ys

