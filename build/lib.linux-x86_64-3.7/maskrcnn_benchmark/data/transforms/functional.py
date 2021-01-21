import cv2
import PIL
import math
import numpy as np

import torch
from torchvision.transforms import functional as TVF
from maskrcnn_benchmark.data.transforms import cvfunctional as CVF


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def _choose_backend(img):
    if TVF._is_pil_image(img):
        return TVF
    elif CVF._is_numpy_image(img):
        return CVF
    else:
        raise Exception("unknown image type {}".format(type(img)))


def _find_backend(img):
    if TVF._is_pil_image(img):
        return 'pil'
    elif CVF._is_numpy_image(img):
        return 'cv2'
    else:
        raise Exception("unknown image type {}".format(type(img)))


# ---------------------------------------------------------------------------- #
# Transform basic functions
# ---------------------------------------------------------------------------- #
def auto_func(img, func_name):
    backend = _choose_backend(img)
    return getattr(backend, func_name)


def hflip(img):
    return auto_func(img, "hflip")(img)


def vflip(img):
    return auto_func(img, "vflip")(img)


def resize(img, size):
    return auto_func(img, "resize")(img, size)


def adjust_brightness(img, brightness_factor):
    return auto_func(img, "adjust_brightness")(img, brightness_factor)


def adjust_contrast(img, contrast_factor):
    """
    warning: behavior is different between PIL and CV2
    """
    return auto_func(img, "adjust_contrast")(img, contrast_factor)


def adjust_saturation(img, saturation_factor):
    """
    warning: behavior is different between PIL and CV2
    """
    return auto_func(img, "adjust_saturation")(img, saturation_factor)


def adjust_hue(img, hue_factor):
    """
    warning: behavior is significantly different between PIL and CV2
    """
    return auto_func(img, "adjust_hue")(img, hue_factor)


def to_tensor(pic):
    return TVF.to_tensor(pic)


def normalize_batch(tensor, mean, std, inplace=False):
    if not inplace:
        tensor = tensor.clone()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def normalize(tensor, mean, std, inplace=False):
    if tensor.ndimension() == 3:
        return TVF.normalize(tensor, mean, std, inplace)
    else:
        return normalize_batch(tensor, mean, std, inplace)


# ---------------------------------------------------------------------------- #
# Util functions
# ---------------------------------------------------------------------------- #
def im_size(img, inverse=False):
    """
    Return the size of the image (width, height) by default. If inverse is True,
    the inverse order (height, width) will be returned
    :param img: PIL.Image or numpy.ndarray object
    :param inverse: bool
    :return: size of the image
    """
    backend = _find_backend(img)
    if backend == "pil":
        s = img.size
    elif backend == "cv2":
        s = (img.shape[1], img.shape[0])
    else:
        raise Exception("unknown backend")
    if inverse:
        s = (s[1], s[0])
    return s


def bounded_normal(lb, ub, mu=0., sigma=1.):
    v = mu + np.random.randn() * sigma
    return np.clip(v, lb, ub)


# ---------------------------------------------------------------------------- #
# Affine transform helper functions
# ---------------------------------------------------------------------------- #
def _get_othogonal_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _rotate_point(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def compute_affine_transform(center, scaled_dim, angle, output_size):
    if isinstance(center, (list, tuple)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scaled_dim, (list, tuple)):
        scaled_dim = np.array(scaled_dim, dtype=np.float32)

    src_w = scaled_dim[0]
    dst_w, dst_h = output_size

    # rotate around the center
    angle = math.radians(angle)
    src_rot_basis = _rotate_point([0, src_w * -0.5], angle)
    dst_rot_basis = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    # add points for transform computation
    src[0, :], src[1, :] = center, center + src_rot_basis
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_rot_basis

    src[2:, :] = _get_othogonal_point(src[0, :], src[1, :])
    dst[2:, :] = _get_othogonal_point(dst[0, :], dst[1, :])

    M, M_inv = cv2.getAffineTransform(np.float32(src), np.float32(dst)), \
               cv2.getAffineTransform(np.float32(dst), np.float32(src))

    return M, M_inv


_resample_fn = {'bilinear': PIL.Image.BILINEAR,
                'nearest': PIL.Image.NEAREST}


def affine_sample(img, trans, inv_trans, dst_size, resample='bilinear'):
    if resample not in _resample_fn:
        raise Exception('unknown resample method [bilinear, nearest]')
    if resample != "bilinear":
        raise NotImplementedError("only bilinear is supported in current version")
    backend = _find_backend(img)
    if backend == "pil":
        img = img.transform(dst_size,
                            PIL.Image.AFFINE,
                            inv_trans.flatten(),
                            resample=_resample_fn[resample])
    elif backend == 'cv2':
        img = cv2.warpAffine(img, trans,
                             dst_size,
                             flags=cv2.INTER_LINEAR)
    else:
        raise Exception("unknown backend")
    return img
