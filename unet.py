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
    def __init__(self, input_shape):
        self.u_history = []
        
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

        conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='conv_10_1')(conv9)
    
        super(UnetModel, self).__init__(inputs=inputs, outputs=conv10, name='unet')
#        self.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')

