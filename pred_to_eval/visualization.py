# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import cv2
import torch
import numpy as np
import os.path as osp
from maskrcnn_benchmark.config import cfg
from torchvision.transforms import functional as F
from maskrcnn_benchmark.data.build import build_dataset
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
# from lib.mitok.utils.mio import mkdir_safe


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class MSVisual(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "lymph_large",
        "coronary_calc",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.min_image_size = min_image_size

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def run_on_opencv_image(self, predictions, image, target):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions: prediction result per image
            target: target per image

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """

        height, width = image.shape[:-1]
        predictions = predictions.resize((width, height))

        if predictions.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = predictions.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [predictions])[0]
            predictions.add_field("mask", masks)
        top_predictions = self.select_top_predictions(predictions)

        result = image[:, :, 3].copy()
        # if self.show_mask_heatmaps:
        #     return self.create_mask_montage(result, top_predictions)

        if self.cfg.MODEL.MASK_ON:
            if len(top_predictions) != 0:
                print("chk1")
                result = self.overlay_mask(result, top_predictions)
            else:
                import pdb; pdb.set_trace()
            if len(target) != 0:
                result = self.overlay_target(result, target)
        else:
            print('Error: mask is not on')
        if len(top_predictions) != 0:
            result = self.overlay_class_names(result, top_predictions)

        return result

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")
        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 1)

        composite = image

        return composite

    def overlay_target(self, image, target):
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

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image


def mkdir_safe(d):
    """
    Make Multi-Directories safety and thread friendly.
    :param d: path
    :return: 0=success, -1=param error
    """
    sub_dirs = d.split('/')
    cur_dir = ''
    max_check_times = 5
    sleep_seconds_per_check = 0.001
    for i in range(len(sub_dirs)):
        cur_dir += sub_dirs[i] + '/'
        for check_iter in range(max_check_times):
            if not os.path.exists(cur_dir):
                try:
                    os.mkdir(cur_dir)
                except:
                    time.sleep(sleep_seconds_per_check)
                    continue
            else:
                break


if __name__ == '__main__':

    prediction_path = '/home/wangcheng/maskrcnn-benchmark/inference/coco_lg3/predictions.pth'
    cfg_path = '/home/wangcheng/maskrcnn-benchmark/ms_mask_rcnn_R_50_FPN_1x_7_1.yaml'
    save_root = '/data2/ms_data/visual_predictions/'

    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    ms_visual = MSVisual(cfg)
    images_predictions = torch.load(prediction_path)
    dataset = build_dataset(cfg, dataset_list=['coco_lg3'], transforms=None,
                            dataset_catalog=DatasetCatalog, is_train=False)[0]

    count = 0
    results = {}
    for i in range(len(dataset)):
        prediction = images_predictions[i]
        pil_image, target_i, idx = dataset[i]
        if idx != i:
            print('index error')
        image_i = np.array(pil_image)
        import pdb; pdb.set_trace()
        visual_image = ms_visual.run_on_opencv_image(prediction, image_i, target_i)

        img_info = dataset.get_img_info(i)
        image_path = osp.join(save_root, img_info['file_name'])
        image_name = image_path.split('/')[-1]
        folder_path = image_path.replace(image_name, '')
        if not osp.exists(osp.dirname(folder_path)):
            mkdir_safe(osp.dirname(folder_path))
        cv2.imwrite(image_path, visual_image)

        print(count)
        count += 1
