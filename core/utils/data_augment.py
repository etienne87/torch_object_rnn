from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
from numpy import random
from core.utils.image import clamp_boxes

def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, labels, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    """
    ar_ = (bbox_area(bbox))

    delta_area = ((ar_ - bbox_area(clip_box)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)


    bbox = bbox[mask == 1, :]
    new_labels = labels[mask == 1]

    return bbox, new_labels

class Compose(object):
    def __init__(self, transfroms):
        self.transfroms = transfroms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transfroms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

        
class LetterBox(object):
    """Resize the image in accordance to 'image_letter_box' function in darknet
    The aspect ratio is maintained."""
    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height

    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        scale_x = float(self.width) / float(width)
        scale_y = float(self.height) / float(height)

        if scale_x > scale_y:
            scale = scale_y
        else:
            scale = scale_x

        new_w = int(width * scale)
        new_h = int(height * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        new_img = np.full((self.height, self.width, 3), 128)

        new_img[(self.height - new_h)//2:(self.height - new_h)//2 + new_h,(self.width - new_w)//2:(self.width - new_w)//2 + new_w, :] = resized_img
        boxes *= scale
        boxes[:, 0] += (self.width - new_w)//2
        boxes[:, 2] += (self.width - new_w)//2
        boxes[:, 1] += (self.height - new_h)//2
        boxes[:, 3] += (self.height - new_h)//2

        return new_img, boxes, labels

class ConvertRGB2HSV(object):
    def __call__(self, img, boxes=None, labels=None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return img, boxes, labels

class ConvertHSV2RGB(object):
    def __call__(self, img, boxes, labels):
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img, boxes, labels

class RandomSaturation(object):
    """Transfrom the image in HSV color space"""
    def __init__(self, saturation=0.5):
        assert saturation > 0.0 and saturation < 1.0
        self.saturation = saturation

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img[:, :, 1] *= random.uniform(-self.saturation, self.saturation)
        return img, boxes, labels

class RandomHue(object):
    """Transfrom the image in HSV color space"""
    def __init__(self, hue=18.0):
        assert hue >= 0.0 and hue <= 360.0
        self.hue = hue

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img[:, :, 0] += random.uniform(-self.hue, self.hue)
            img[:, :, 0] = np.clip(img[:, :, 0], 0, 360)
        return img, boxes, labels

class RandomBrightness(object):
    """Tranfrom the image in RGB color space"""
    def __init__(self, brightness=32):
        assert brightness > 0.0 and brightness < 255.0
        self.brightness = brightness

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img += random.uniform(-self.brightness, self.brightness)
            img = np.clip(img, 0.0, 255.0)
        return img, boxes, labels

class RandomContrast(object):
    """Tranfrom the image in RGB color space"""
    def __init__(self, low=0.5, high=0.99):
        self.low = low
        self.high = high

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            img *= random.uniform(self.low, self.high)
            img = np.clip(img, 0.0, 255.0)
        return img, boxes, labels


class DictWrapper(object):
    def __init__(self, da):
      self.da = da

    def __call__(self, sample):
        image = sample['img']
        boxes = sample['annot']
        labels = boxes[:, 4]
        boxes = boxes[:, :4]
        image, boxes, labels = self.da(image, boxes, labels)
        boxes = clamp_boxes(boxes, image.shape[0], image.shape[1])
        sample['img'] = image
        sample['annot'] = np.concatenate((boxes, labels[:, None]), axis=1)
        return sample


class PhotometricDistort(object):
    def __init__(self):
        self.pd = Compose([
            RandomContrast(),
            ConvertRGB2HSV(),
            RandomSaturation(),
            RandomHue(),
            ConvertHSV2RGB(),
            RandomBrightness()
        ])

    def __call__(self, img, boxes=None, labels=None):
        img, boxes, labels = self.pd(img * 255.0, boxes, labels)
        return img/ 255.0, boxes, labels

class AddGaussNoise(object):
    def __init__(self, sigma=2):
        self.sigma = sigma

    def __call__(self, img, boxes=None, labels=None):
        sigma = random.uniform(0, self.sigma)
        img = cv2.GaussianBlur(img, (5,5), sigmaX=sigma)
        return img, boxes, labels

class Expand(object):
    """Expand the image
    mean: the pixel value of the expand area.
    ratio: the max ratio of the orgin image width/height with expanded image width.height """
    def __init__(self, mean=128, ratio=2):
        self.ratio = ratio
        self.mean = mean

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            return img, boxes, labels

        ratio = random.uniform(1, self.ratio)
        height, width, channels = img.shape
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        left = random.randint(0, new_w - width)
        top = random.randint(0, new_h - height)
        expand_image = np.full((new_h, new_w, 3), self.mean)
        expand_image[top:top+height, left:left+width, :] = img
        img = expand_image

        boxes[:, 0] += left
        boxes[:, 2] += left
        boxes[:, 1] += top
        boxes[:, 3] += top

        return img, boxes, labels


class Four_Point_Crop(object):
    """crop_x and crop_y  range from -0.5 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    crop_x > 0 , crop_y > 0 crop image upper left part
    crop_x > 0 , crop_y < 0 crop image Lower left part
    crop_x < 0 , crop_y > 0 crop image upper right part
    crop_x < 0 , crop_y < 0 crop image right lower part
    """
    def __init__(self, crop_x=0.2, crop_y=0.2):
        assert crop_x < 0.5 and crop_x > -0.5
        assert crop_y < 0.5 and crop_y > -0.5

        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        new_w = int(width * (1 - abs(self.crop_x)))
        new_h = int(height * (1 - abs(self.crop_y)))
        [left, top, right, bottom] = [0, 0, width, height]
        if self.crop_x >= 0 and self.crop_y >= 0:
            right = new_w
            bottom = new_h

        if self.crop_x <= 0 and self.crop_y <= 0:
            left = width - new_w
            top = height - new_h

        if self.crop_x >=0 and self.crop_y <= 0:
            top = height -new_h
            right = new_w

        if self.crop_x <= 0 and self.crop_y >= 0:
            left = width - new_w
            bottom = new_h
        new_img = img[top:bottom, left:right, :]

        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], left)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], top)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], right)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top
        return new_img, boxes, labels

class CenterCrop(object):
    """crop_x and crop_y  range from 0 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, crop_x=0.25, crop_y=0.25):
        assert crop_x < 0.5 and crop_x > 0
        assert crop_y < 0.5 and crop_y > 0

        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        new_w = int(width * (1 - abs(self.crop_x)))
        new_h = int(height * (1 - abs(self.crop_y)))
        left = (width - new_w) // 2
        top  = (height -new_h) // 2
        right = left + new_w
        bottom = top + new_h

        new_img = img[top:bottom, left:right, :]
        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], left)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], top)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], right)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top
        return new_img, boxes, labels

class RandomFour_Point_Crop(object):
    """crop_x and crop_y  range from 0 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, crop_x=0.2, crop_y=0.2):
        assert crop_x < 0.5 and crop_x > 0
        assert crop_y < 0.5 and crop_y > 0

        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        crop_x = random.uniform(-self.crop_x, self.crop_x)
        crop_y = random.uniform(-self.crop_y, self.crop_y)
        height, width, channels = img.shape
        new_w = int(width * (1 - abs(crop_x)))
        new_h = int(height * (1 - abs(crop_y)))
        [left, top, right, bottom] = [0, 0, width, height]
        if crop_x >= 0 and crop_y >= 0:
            right = new_w
            bottom = new_h

        if crop_x <= 0 and crop_y <= 0:
            left = width - new_w
            top = height - new_h

        if crop_x >=0 and crop_y <= 0:
            top = height -new_h
            right = new_w

        if crop_x <= 0 and crop_y >= 0:
            left = width - new_w
            bottom = new_h
        new_img = img[top:bottom, left:right, :]

        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], left)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], top)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], right)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top

        return new_img, boxes, labels

class RandomCenterCrop(object):
    """crop_x and crop_y  range from 0 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, crop_x=0.49, crop_y=0.49):
        assert crop_x < 0.5 and crop_x > 0
        assert crop_y < 0.5 and crop_y > 0

        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        crop_x = random.uniform(-self.crop_x, self.crop_x)
        crop_y = random.uniform(-self.crop_y, self.crop_y)
        height, width, channels = img.shape
        new_w = int(width * (1 - abs(crop_x)))
        new_h = int(height * (1 - abs(crop_y)))
        left = (width - new_w) // 2
        top  = (height -new_h) // 2
        right = left + new_w
        bottom = top + new_h

        new_img = img[top:bottom, left:right, :]
        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], left)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], top)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], right)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top
        return new_img, boxes, labels

class RandomCrop(object):
    """crop_x and crop_y  range from 0 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, crop_x=0.3, crop_y=0.3):
        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(0, 5) == 0:
            randomcrop = Compose([RandomCenterCrop(self.crop_x, self.crop_y)])
        else:
            randomcrop = Compose([RandomFour_Point_Crop(self.crop_x, self.crop_y)])
        img, boxes, labels = randomcrop(img, boxes, labels)
        return img, boxes, labels


def main():
    imgname = ''
    boxes =[]
    labels = []
    img = cv2.imread(imgname)
    dataAug = Augmentation()
    auged_img, auged_bboxes, auged_labels = dataAug(img, boxes, labels)

if __name__ == "__main__":
    main()