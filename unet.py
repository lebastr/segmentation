import numpy as np
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, Reshape,
                          BatchNormalization, Concatenate)

import keras.applications.vgg16

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

def create_VGG16_conv_pipe(img_input):
    # Block 1
    net = {}

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    net['b1'] = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    net['b2'] = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    net['b3'] = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    net['b4'] = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    net['to-unet'] = x
    return net

class VGGUnetModel(Model):
    def __init_vgg_unet__(self, labels=2):
        def two_conv2d(name, n_filter1, n_filter2, x):
            x = Conv2D(n_filter1, (3, 3), activation='relu', padding='same', name=name + "_1")(x)
            x = Conv2D(n_filter2, (3, 3), activation='relu', padding='same', name=name + "_2")(x)
            return x

        def conv_concate_block(name, n_filter, x, y):
            x = BatchNormalization(name=name+"batch_norm_x")(x)
            y = BatchNormalization(name=name+"batch_norm_y")(y)
            x = two_conv2d(name + "two_conv_block", n_filter, n_filter // 2, x)
            return Concatenate()([UpSampling2D(size=(2, 2))(x), y])

        #vgg = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None,
        #                                     input_shape=None, pooling=None, classes=1000)

        input_img = Input(shape=(224, 224, 3))
        vgg_pipe = create_VGG16_conv_pipe(BatchNormalization(name="input_batch_norm")(input_img))

        x = conv_concate_block("cc1", 1024, vgg_pipe['to-unet'], vgg_pipe['b4'])
        x = conv_concate_block("cc2", 512, x, vgg_pipe['b3'])
        x = conv_concate_block("cc3", 256, x, vgg_pipe['b2'])
        x = conv_concate_block("cc4", 128, x, vgg_pipe['b1'])

        x = two_conv2d("last_conv_block", 64, 64, x)

        x = Conv2D(labels, (1, 1), activation='sigmoid', padding='same', name="sigmoid_conv")(x)

        super(VGGUnetModel, self).__init__(inputs=input_img, outputs=x, name='vgg-unet')
        self.vgg_layers = [self.layers[i] for i in range(2,15)]

    def __init__(self):
        self.u_history = []
        self.__init_vgg_unet__()

    def load_vgg_weights(self):
        vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None,
                                             input_shape=None, pooling=None, classes=1000)

        for i, layer in enumerate(vgg.layers[1:14]):
            self.vgg_layers[i].set_weights(layer.get_weights())
