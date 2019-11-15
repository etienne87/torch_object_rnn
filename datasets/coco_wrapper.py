from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from pycocotools.coco import COCO
import json
from tqdm import tqdm

def coco_split(root_dir, set_name):
    """COCO takes a lot of memory, this dumps a pickled list of filenames & annotations as jsons
    """
    path = os.path.join(root_dir, 'annotations', 'instances_' + set_name + '.json')
    out_folder = os.path.join(root_dir, 'annotations', 'instances_' + set_name)
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    coco = COCO(path)
    image_ids = coco.getImgIds()
    dataset = {'image_ids':image_ids, 
                'image_id_to_annot_path':{}, 
                'image_id_to_image_info': {},
                'categories': coco.loadCats(coco.getCatIds()),
                }
    for image_index in tqdm(image_ids, total=len(image_ids)):
        annotations_ids = coco.getAnnIds(imgIds=image_index, iscrowd=False)
        coco_annotations = coco.loadAnns(annotations_ids)
        image_info = coco.loadImgs(image_index)[0]
        annotation_path = os.path.join(out_folder, 'annot#'+str(image_index)+'.json')

        image_index_t = str(image_index)
        dataset['image_id_to_image_info'][image_index_t] = image_info
        dataset['image_id_to_annot_path'][image_index_t] = annotation_path
        with open(annotation_path, 'w') as fp:
            json.dump(coco_annotations, fp) 

    dataset_path = os.path.join(out_folder, 'dataset.json')
    with open(dataset_path, 'w') as fp:
            json.dump(dataset, fp)

    #small test
    with open(dataset_path, 'r') as fp:
        test = json.load(fp)
    for key in test.keys():
        assert test[key] == dataset[key]
    

class COCO1(object):
    def __init__(self, root_dir, set_name):
        """wrapper around COCO to test memory leak"""
        self.root_dir = root_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
    
    def get_image_ids(self):
        return self.coco.getImgIds()
        
    def load_categories(self):
        return self.coco.loadCats(self.coco.getCatIds())

    def load_image(self, image_id):
        return self.coco.loadImgs(image_id)[0]

    def load_annotation(self, image_id):
        annotations_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        annotations = np.zeros((0, 5))
        if len(annotations_ids) == 0:
            return annotations
        return self.coco.loadAnns(annotations_ids)


class COCO2(object):
    def __init__(self, root_dir, set_name):
        """COCO memory is leaking & takes too much memory, this class dumps/ loads annotation file per image
           is initialized with real coco object
        """
        self.root_dir = root_dir
        self.set_name = set_name 
        data_filename = os.path.join(root_dir, 'annotations', 'instances_' + set_name, 'dataset.json')
        if not os.path.exists(data_filename):
            print('preparing coco annotations...')
            coco_split(root_dir, set_name)
        with open(data_filename, 'r') as fp:
            data_dict = json.load(fp)
        self.__dict__.update(data_dict)
    
    def get_image_ids(self):
        return self.image_ids
        
    def load_categories(self):
        return self.categories

    def load_image(self, image_id):
        return self.image_id_to_image_info[str(image_id)]

    def load_annotation(self, image_id):
        path = self.image_id_to_annot_path[str(image_id)]
        with open(path, 'r') as fp:
            annotation = json.load(fp)
        
        annotations = np.zeros((0, 5))
        if len(annotation) == 0:
            return annotations
        return annotation