import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

import os
import os.path
import re
from matplotlib.collections import PolyCollection

import PIL
from PIL import ImageDraw
from PIL import Image as PILImage

import json
import random

ChannelMUL = "MUL"
ChannelMUL_PanSharpen = "MUL-PanSharpen"
ChannelPAN = "PAN"
ChannelRGB_PanSharpen = "RGB-PanSharpen"
        
IMAGE_SIZE = (650, 650)

class Building(object):
    def __init__(self, image_id, building_id, poly_array):
        self.image_id = image_id
        self.building_id = building_id
        assert len(poly_array) > 0, "Poly_array must not be empty"
        assert len(poly_array) <= 2, "Poly_array must consist no more 2 polygons"
        self.outer_poly = poly_array[0]
        self.inner_poly = poly_array[1] if len(poly_array) == 2 else []

def apply_pil_transform(f, X):
    Xs_new = []
    for j in range(X.shape[2]):
        img = PILImage.fromarray(X[:,:,j])
        Xs_new.append(np.array(f(img)))

    Xs_new = np.array(Xs_new)
    return Xs_new.transpose([1,2,0])

def rotate_and_translate_crop(X, angle, translate=None, crop=0):
    X_new = np.empty(X.shape)
    for j in range(X.shape[2]):
        img = PILImage.fromarray(X[:,:,j]).rotate(angle, translate=translate)
        if crop > 0:
            img = img.crop((crop, crop, X.shape[0] - crop, X.shape[1] - crop))
        X_new[:,:,j] = np.array(img)

    return X_new    
    

class Image(object):
    def __init__(self, data_dir, image_id):
        self.data_dir = data_dir
        self.image_id = image_id


    def get_ndarray(self, channels, image_size=None):
        imgs = []
        ch_psums = [0]

        if image_size is None:
            image_size = IMAGE_SIZE
        
        for ch in channels:
            path = self.data_dir + "/" + ch
            img = np.float32(tiff.imread("%s/%s.tif" % (path, self.image_id)))

            if len(img.shape) == 2:
                img = img[:,:,None]

            assert img.shape[0:2] == IMAGE_SIZE, "image size must be size %d x %d, but has %d x %d" % (IMAGE_SIZE[0], IMAGE_SIZE[1], img.shape[0], img.shape[1])

            if img.shape[0:2] != image_size:
                img = apply_pil_transform(lambda i: i.resize(image_size, resample=PILImage.BILINEAR), img)

            ch_psums.append(ch_psums[-1] + img.shape[2])
            imgs.append(img)
        
        image = np.zeros((image_size[0], image_size[1], ch_psums[-1]), dtype='float32')
        for i in range(len(imgs)):
            image[:,:,ch_psums[i]:ch_psums[i+1]] = imgs[i]
        
        return image
                
    def draw(self, ax=None):
        img = self.get_ndarray([ChannelRGB_PanSharpen])
        img = np.uint8(256 * np.float32(img) / img.max())
        
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.imshow(img, alpha=1.0)
        
class ImageWithBuildings(Image):
    def __init__(self, data_dir, image_id, buildings):
        super(ImageWithBuildings, self).__init__(data_dir, image_id)
        self.buildings = buildings
        
    def get_mask(self, image_size=None, only_border=False):
        poly = PILImage.new('L', IMAGE_SIZE)
        pdraw = ImageDraw.Draw(poly)
        for p_out, p_in in [(b.outer_poly, b.inner_poly) for b in self.buildings]:
            if only_border:
                pdraw.line(map(tuple, p_out), fill=1, width=5)
            else:
                pdraw.polygon(map(tuple, p_out), fill=1,outline=0)
                
            if len(p_in) != 0:
                pdraw.polygon(map(tuple, p_in), fill=0,outline=0)

        if image_size is not None:
            poly = poly.resize(image_size, resample=PILImage.BILINEAR)

        return np.array(poly, dtype='float32')
    
    def draw(self, ax=None, withbuildings=True):
        if ax is None:
            fig, ax = plt.subplots()
        
        super(ImageWithBuildings, self).draw(ax)
        if withbuildings:
            coll_out = PolyCollection([b.outer_poly for b in self.buildings], edgecolors='#ff0000', alpha=0.3)
            ax.add_collection(coll_out)
        
            coll_in = PolyCollection([b.inner_poly for b in self.buildings if len(b.inner_poly) > 0], edgecolors='#ffff00', alpha=0.3)
            ax.add_collection(coll_in)
        
def parse_csv(fname):
    def parse_line(s):
        s1 = s.split('"')
        image_id, building_id, _ = s1[1].split(',')
        poly_array = [[[float(x) for x in vs.split(' ')[0:2]] 
                            for vs in xs[1:-1].split(",")] 
                                for xs in re.findall(r'\([^()]+\)', s1[3][9:-1])]
        return image_id, building_id, poly_array
        
    fd = open(fname, 'r')
    csv = []
    for j,l in list(enumerate(fd.readlines()))[1:]:
        try:
            row = parse_line(l)
        except BaseException as e:
            print ("line: %d, text: %s" % (j,l))
            raise e
        csv.append(row)
    return csv

def load_data_set_from_dumps(strjson):
    js = json.loads(strjson)
    return DataSet(js['data_dir'], js['channels'], image_size=js['image_size'], only_border=js['only_border'])

class DataSet(object):
    def __init__(self, data_dir, channels, image_size=None, only_border=False):
        self.data_dir = data_dir
        self.channels = channels
        self.image_size = image_size

        self.only_border = only_border
        self.load()
        
    def dumps(self):
        return json.dumps({'data_dir' : self.data_dir,
                           'channels' : self.channels,
                           'image_size' : self.image_size,
                           'only_border' : self.only_border})

    def image_ids(self):
        return self.images.keys()

    def get_ndarray(self, image_id):
        return self.images[image_id].get_ndarray(self.channels, self.image_size)

    def get_mask(self, image_id):
        return self.images[image_id].get_mask(self.image_size, only_border=self.only_border)

    def draw(self, image_id, ax=None, withbuildings = True):
        self.images[image_id].draw(ax=ax, withbuildings=withbuildings)
    
    def load(self):
        # read tiff files list
        def get_tif_list():
            path = self.data_dir + "/" + ChannelPAN
            return [fname[:-4] 
                      for (fname, full_path) 
                      in [(fname, os.path.join(path, fname)) 
                           for fname 
                           in os.listdir(path)] 
                    if os.path.isfile(full_path) and fname[-3:] == 'tif']
    
        images = {}
        for t in get_tif_list():
            images[t] = [] # ImageWithBuildings(self.data_dir, [])
        
        # parse csv file and load building polygons
        def load_buildings():
            fname = self.data_dir + "/summaryData/Train_Set.csv"
            return [Building(row[0], row[1], row[2]) for row in parse_csv(fname)]

        for b in load_buildings():
            images[b.image_id].append(b)
        
        for image_id in images.keys():
            images[image_id] = ImageWithBuildings(self.data_dir, image_id, images[image_id])
            
        self.images = images

        self.shape = self.get_ndarray(self.image_ids()[0]).shape

        # load test and train
        train_test_path = self.data_dir + "/train_test.json"
        if not os.path.exists(train_test_path):
            print("Create train and test datasets...")
            self.__split_on_train_test__(train_test_path)

        self.train_ids, self.test_ids = self.__load_train_test__(train_test_path)
        
        return images.keys()

    def __load_train_test__(self, path):
        d = json.load(open(path, "r"))
        return d['train'], d['test']
        
    def __split_on_train_test__(self, path):
        image_ids = self.image_ids()
        random.shuffle(image_ids)
        split_point = len(image_ids)*7 // 10
        train_ids = image_ids[:split_point]
        test_ids = image_ids[split_point:]
        json.dump({'train' : train_ids, 'test' : test_ids}, open(path, "w"))
