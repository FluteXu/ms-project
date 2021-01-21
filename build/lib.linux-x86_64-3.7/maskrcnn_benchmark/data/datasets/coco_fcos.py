# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
#import torchvision

import sys
import os
import cv2
from PIL import Image
# modified
sys.path.append('/home/wangcheng/FcosNet')

from maskrcnn_benchmark.structures.bounding_box import BoxList
#from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
#from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class CocoDetection(object):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, annFile, slice_range):
        super(CocoDetection, self).__init__()
        from cocoapi.PythonAPI.pycocotools.coco import COCO
        self.root = root
        self.range = slice_range
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        #print('ann_ids:', ann_ids)
        target = coco.loadAnns(ann_ids)
        #print('target:', target)

        path = coco.loadImgs(img_id)[0]['file_name']
        #print(path)

        # modified （add for lung detection)
        path_split = path.split('/')
        sub_dir = path[:-len(path_split[-1])]
        slice_index = int(path_split[-1][:-4]) # current slice index

        '''
        image_merge = []
        half_range = self.range // 2
        for i in range(self.range):
            merge_slice_index = slice_index + i - half_range
            merge_slice_path = os.path.join(self.root, sub_dir, '%03d.png'%merge_slice_index)
            if not os.path.exists(merge_slice_path):
                merge_slice_path = os.path.join(self.root, sub_dir, '%03d.png'%slice_index)
            image_merge.append(Image.open(merge_slice_path))
        '''

        #print(slice_index)
        slice_index_before = slice_index - 1
        slice_index_next = slice_index + 1

        slice_before_path = os.path.join(self.root, sub_dir, '%03d.png'%slice_index_before)
        slice_next_path = os.path.join(self.root, sub_dir, '%03d.png'%slice_index_next)
        if not os.path.exists(slice_before_path):
            slice_before_path = os.path.join(self.root, sub_dir, '%03d.png'%slice_index)
        if not os.path.exists(slice_next_path):
            slice_next_path = os.path.join(self.root, sub_dir, '%03d.png'%slice_index)

        #print('slice_before_path:', slice_before_path)
        #print('middle_path:', os.path.join(self.root, path))
        #print('slice_next_path:', slice_next_path)
        img_up = Image.open(slice_before_path)
        img_middle = Image.open(os.path.join(self.root, path))
        img_bottom = Image.open(slice_next_path)

        img = Image.merge('RGB', [img_up, img_middle, img_bottom])


        # modified
        #img = Image.open(os.path.join(self.root, path)).convert('RGB')

        return img, target, os.path.join(self.root, path)


    def __len__(self):
        return len(self.ids)

class COCODataset(CocoDetection):
#class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        # modified
        #self, ann_file, root, remove_images_without_annotations, transforms=None, expands=None
        self, ann_file, root, slice_range, transforms=None, expands=None
    ):
        super(COCODataset, self).__init__(root, ann_file, slice_range)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        #print('success:', slice_range)

        '''
        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        '''

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.expands = expands

    def __getitem__(self, idx):
        img, anno, img_path = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy") # convert xywh format to xyxy format

        # modified (add expands to target)
        if self.expands != [] and self.expands is not None:
            #print('success')
            target = target.expand_bbox_by_pix(expand=self.expands[0], size_thresh=self.expands[1])
        #target = target.expand_bbox_by_pix(3, 16)


        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        #print('class:', classes)
        target.add_field("labels", classes)

        # modified
        '''
        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)
        '''

        #print('before clip: ', target.bbox)
        #print('before_clip_path: ', img_path)

        #print('target before:', target)
        target = target.clip_to_image(remove_empty=True) # make sure bbox is inside the image
        #print('target after:', target)

        #print('before: ', target.bbox)
        #print('before_path: ', img_path)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        #print('trans: ', target.bbox)
        #print('path: ', img_path)

        return img, target, img_path, idx

    # get image's information:
    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
