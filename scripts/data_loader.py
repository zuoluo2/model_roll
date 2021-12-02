import os
import tensorflow as tf
import argparse
import scipy
import scipy.misc
import random
from PIL import Image
from PIL import ImageEnhance
import numpy.random as npr

# data augmentation
ADJUST_BRIGHTNESS = False
ADJUST_CONTRAST = True

def adjust_brightness(img):
    enhancer = ImageEnhance.Brightness(img)
    factor = npr.uniform(0.5, 1.5)
    return enhancer.enhance(factor)

def adjust_contrast(img):
    enhancer = ImageEnhance.Contrast(img)
    factor = npr.uniform(1., 1.5)
    if npr.random() < 0.5:
        factor = 1./factor
    return enhancer.enhance(factor)

class DataLoader:
    def __init__(self, dataset_index_path, data_file, up_crop):
        self.up_crop = up_crop
        self.train_bags = []
        if data_file:
            self.dataset_index = [data_file]
        else:
            self.dataset_index_path = dataset_index_path
            self.dataset_index = []
            with open(self.dataset_index_path, 'r') as dataset_index_file:
                for line in dataset_index_file:
                    if not line or line[0] == '#':
                        continue
                    dataset_path = line.rstrip('\n')
                    self.dataset_index.append(dataset_path)

    def load_for_train(self, skip_frames):
        self.skip_frames = skip_frames
        self.load(True)
    def load_for_eval(self):
        self.skip_frames = 0
        self.load(False)
        
    def load(self, for_train):
        xs = []
        ys = []
        
        for dataset_path in self.dataset_index:
            print "Loading dataset ", dataset_path
            last_train_frame = 0
            last_frame = 0
            bag_xs = []
            bag_ys = []
            bag_curr_roll = []
            with open(os.path.join(dataset_path, 'result.tsv'), 'r') as f:
                for line in f:
                    last_frame += 1
                    if last_frame - last_train_frame <= self.skip_frames:
                        continue
                    fs = line.rstrip('\n').split('\t')
                    ts = fs[0]
                    index = fs[1]
                    image_name = ts + ('_%05d' % int(index))
                    label = float(fs[4])
                    curr_roll = float(fs[2])
                    weight = 1.0
                    if for_train and abs(label - curr_roll) > 0.5:
                        weight = 4.0
                    for i in range(int(weight)):
                        bag_xs.append(os.path.join(dataset_path, 'raw', image_name + '.jpg'))
                        bag_ys.append(label)
                        bag_curr_roll.append(curr_roll)
                    last_train_frame = last_frame

                    # do data augmentation to training set
                    if for_train and ADJUST_BRIGHTNESS:
                        if not os.path.exists(dataset_path + '/brightened'):
                            os.makedirs(dataset_path + '/brightened')
                        image = Image.open(os.path.join(dataset_path, 'raw', image_name + '.jpg'))
                        brightened_img = adjust_brightness(image)
                        brightened_img.save(dataset_path + '/brightened/' + image_name + '.jpg')
                        bag_xs.append(os.path.join(dataset_path, 'brightened', image_name + '.jpg'))
                        bag_ys.append(label)
                        bag_curr_roll.append(curr_roll)
                    if for_train and ADJUST_CONTRAST:
                        if not os.path.exists(dataset_path + '/contrast'):
                            os.makedirs(dataset_path + '/contrast')
                        image = Image.open(os.path.join(dataset_path, 'raw', image_name + '.jpg'))
                        contrast_img = adjust_contrast(image)
                        contrast_img.save(dataset_path + '/contrast/' + image_name + '.jpg')
                        bag_xs.append(os.path.join(dataset_path, 'contrast', image_name + '.jpg'))
                        bag_ys.append(label)
                        bag_curr_roll.append(curr_roll)
            self.train_bags.append([dataset_path, bag_xs, bag_ys, bag_curr_roll])
            xs += bag_xs
            ys += bag_ys

        print "Total data: ", len(xs)
        c = list(zip(xs, ys))
        if for_train:
            random.shuffle(c)
        xs, ys = zip(*c)
        self.train_xs = xs
        self.train_ys = ys
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0
        
    def load_train_batch(self, batch_size):
        x_out = []
        y_out = []
        x_name = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.train_xs[(self.train_batch_pointer + i) % len(self.train_xs)])
            # x_out.append(scipy.misc.imresize(image, [66, 200]) / 255.0)
            x_out.append(scipy.misc.imresize(image[self.up_crop:, :], [270, 480]) / 255.0)
            # scipy.misc.imsave('test'+str(i) + '.png', x_out[-1])
            # x_out.append(image)
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % len(self.train_xs)]])
            x_name.append(self.train_xs[(self.train_batch_pointer + i) % len(self.train_xs)])
        self.train_batch_pointer += batch_size
        return x_out, y_out, x_name

    def load_eval_batch(self, samples, index):
        x_out = []
        y_out = []
        x_name = []
        xs = samples[1]
        ys = samples[2]
        curr_rolls = samples[3]
        image = scipy.misc.imread(xs[index])
        # x_out.append(scipy.misc.imresize(image, [66, 200]) / 255.0)
        x_out.append(scipy.misc.imresize(image[self.up_crop:, :], [270, 480]) / 255.0)
        # x_out.append(image)
        y_out.append([ys[index]])
        x_name.append(xs[index])
        
        return x_out, y_out, x_name, [curr_rolls[index]]
    
