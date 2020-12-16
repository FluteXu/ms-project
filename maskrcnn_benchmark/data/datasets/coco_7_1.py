# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
# import torchvision

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.utils import cv2_util

import os
import re
import sys
import cv2
import os.path
import numpy as np
from PIL import Image
import torch.utils.data as data
sys.path.append('/home/wangcheng/maskrcnn-benchmark/cocoapi/PythonAPI')

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

# ============================ visualization =============================

def overlay_target(image, target):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        target (SegmentationMask): draw ground truth with fixed color
    """

    masks = target.get_field("masks")
    for mask in masks:
        mask = mask.get_mask_tensor()
        contours, hierarchy = cv2_util.findContours(
            np.array(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # draw targets with fixed colors
        image = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    composite = image

    return composite


def coco_to_ann_cnts(contours):
    cnts = []
    for contour in contours:
        cnt = []
        for j in range(int(len(contour)/2)):
            cnt.append([contour[j*2], contour[j*2+1]])
        cnts.append(cnt)

    return np.array(cnts)


def visualize_one_roi(image, target):
        for mask in target:
            # draw targets with fixed colors
            contour = np.array(mask['segmentation']).astype(int)
            contour = coco_to_ann_cnts(contour)
            print(contour)
            image = cv2.drawContours(image, contour, -1, (0, 255, 0), 1)
        composite = image

        return composite


def cv2_imshow(img, target):
    image = np.array(img)
    # image is (x, y, z) i.e.(512, 512, 3)
    # image = np.transpose(image, [2, 0, 1])
    print('shape: ', image.shape)
    cv2.imwrite('0.png', image[2])
    composite = overlay_target(image[3], target)
    cv2.imwrite('1.png', composite)
    cv2.imwrite('2.png', image[4])


def _load_image_cv2(root, path):
    n_neigh = cfg.INPUT.SLICE_NUM
    sub_dir = os.path.dirname(path)
    cent_slice = int(re.findall(re.compile(r'/(\d+).png'), path)[0])
    assert n_neigh % 2 == 1, "#input slices should be odd"
    half_n_neigh = n_neigh // 2

    paths = [[], []]
    for offset in range(half_n_neigh+1):
        for sign, container in zip([-1, 1], paths):
            slice_id = cent_slice + offset * sign
            fpath = os.path.join(root, sub_dir, '{:03d}.png'.format(slice_id))
            if not os.path.exists(fpath):
                fpath = container[-1]
            container.append(fpath)
    paths = paths[0][::-1] + paths[1][1:]
    channels = [cv2.imread(fpath, cv2.IMREAD_UNCHANGED) for fpath in paths]
    return np.stack(channels, axis=2)


def _load_image_cv2_padding(root, path):
    n_neigh = cfg.INPUT.SLICE_NUM
    sub_dir = os.path.dirname(path)
    cent_slice = int(re.findall(re.compile(r'/(\d+).png'), path)[0])
    assert n_neigh % 2 == 1, "#input slices should be odd"
    half_n_neigh = n_neigh // 2

    paths = [[], []]
    for offset in range(half_n_neigh + 1):
        for sign, container in zip([-1, 1], paths):
            slice_id = cent_slice + offset * sign
            fpath = os.path.join(root, sub_dir, '{:03d}.png'.format(slice_id))
            if not os.path.exists(fpath):
                fpath = None
            container.append(fpath)
    paths = paths[0][::-1] + paths[1][1:]
    # channels = [cv2.imread(fpath, cv2.IMREAD_UNCHANGED) for fpath in paths]

    channels = []
    shape = cv2.imread(paths[len(paths) // 2], cv2.IMREAD_UNCHANGED).shape
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    for fpath in paths:
        if fpath is None:
            this_img = np.zeros(shape)
        else:
            this_img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        channels.append(np.stack([this_img, X, Y], axis=2))
    return np.concatenate(channels, axis=2)


class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.zflip_prob = 0.5

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
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = _load_image_cv2(self.root, path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class COCODataset(CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.shape[:-1], mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.shape[:-1], mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.shape[:-1])
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

            # visualize merged channel
            # cv2_imshow(img, target)
            # import pdb; pdb.set_trace()

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


