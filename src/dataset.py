import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

import os
import os.path
import re
from matplotlib.collections import PolyCollection

from PIL import ImageDraw
from PIL import Image as PILImage

import json
import random

DATA_DIR = '../dataset/trainset/'
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

class Image(object):
    def __init__(self, data_dir, image_id):
        self.data_dir = data_dir
        self.image_id = image_id


    def get_ndarray(self, channels):
        imgs = []
        ch_psums = [0]

        for ch in channels:
            path = self.data_dir + "/" + ch
            img = np.float32(tiff.imread("%s/%s.tif" % (path, self.image_id)))

            if len(img.shape) == 2:
                img = img[None,:,:]
            else:
                img = img.transpose([2,0,1])

            assert img.shape[1:] == IMAGE_SIZE, "image size must be size %d x %d, but has %d x %d" % (IMAGE_SIZE[0], IMAGE_SIZE[1], img.shape[1], img.shape[2])

            ch_psums.append(ch_psums[-1] + img.shape[0])
            imgs.append(img)
        
        image = np.zeros((ch_psums[-1], IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype='float32')
        for i in range(len(imgs)):
            image[ch_psums[i]:ch_psums[i+1], :, :] = imgs[i]
        
        return image
                
    def draw(self, ax=None):
        img = self.get_ndarray([ChannelRGB_PanSharpen]).transpose([1,2,0])
        img = np.uint8(256 * np.float32(img) / img.max())
        
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.imshow(img, alpha=1.0)
        
class ImageWithBuildings(Image):
    def __init__(self, data_dir, image_id, buildings):
        super(ImageWithBuildings, self).__init__(data_dir, image_id)
        self.buildings = buildings

    def get_interior_mask(self):
        mask = PILImage.new('L', IMAGE_SIZE)
        canvas = ImageDraw.Draw(mask)
        for p_out, p_in in [(b.outer_poly, b.inner_poly) for b in self.buildings]:
            canvas.polygon(map(tuple, p_out), fill=1, outline=0)
            if len(p_in) != 0:
                canvas.polygon(map(tuple, p_in), fill=0, outline=0)

        return np.array(mask, dtype='uint32')[None,:,:]

    def get_border_mask(self, width = 5):
        mask = PILImage.new('L', IMAGE_SIZE)
        canvas = ImageDraw.Draw(mask)
        for p_out, p_in in [(b.outer_poly, b.inner_poly) for b in self.buildings]:
            canvas.line(map(tuple, p_out), fill=1, width=width)

        return np.array(mask, dtype='uint32')[None,:,:]

    def get_corner_mask(self, radius=5):
        mask = PILImage.new('L', IMAGE_SIZE)
        canvas = ImageDraw.Draw(mask)
        for p_out, p_in in [(b.outer_poly, b.inner_poly) for b in self.buildings]:
            for x, y in p_out:
                canvas.ellipse([x - radius, y - radius, x + radius, y + radius], fill=1, outline=1)

        return np.array(mask, dtype='uint32')[None,:,:]

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

class DataSet(object):
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.load()

        
    def image_ids(self):
        return self.images.keys()

    def get_image(self, im_id):
        return self.images[im_id]

    def train_images(self):
        return [self.images[k] for k in self.train_ids]

    def test_images(self):
        return [self.images[k] for k in self.test_ids]

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

        # load test and train
        train_test_path = self.data_dir + "/train_test.json"
        if not os.path.exists(train_test_path):
            print("Create train and test datasets...")
            self.__split_on_train_test__(train_test_path)

        self.train_ids, self.test_ids = self.__load_train_test__(train_test_path)
        
        return images.keys()

    @staticmethod
    def __load_train_test__(path):
        d = json.load(open(path, "r"))
        return d['train'], d['test']
        
    def __split_on_train_test__(self, path):
        image_ids = self.image_ids()
        random.shuffle(image_ids)
        split_point = len(image_ids)*7 // 10
        train_ids = image_ids[:split_point]
        test_ids = image_ids[split_point:]
        json.dump({'train' : train_ids, 'test' : test_ids}, open(path, "w"))