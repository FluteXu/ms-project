# -*- coding: utf-8 -*-
import os
import json
import sys
import cv2
import nrrd
import numpy
import math
import uuid
import torch
import numpy as np
import os.path as osp
from itertools import compress
# from diameter import SegSlice
from collections import namedtuple
from maskrcnn_benchmark.utils import cv2_util
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------------------------
# JSON CLASS
# ---------------------------------------------------------------------------------------------
class StandardJSON(object):
    """
    load, write json files and generate standard json 
    """

    def load_json(self, json_file):
        """
        load json file in a safe manner
        :return: a dict
        :author: zzhou
        :date: 2018-06-06
        """
        json_string = self.delete_dot_mark(json_file, is_overwrite=False)[0]
        json_dict = eval(json_string)
        return json_dict

    def write_json(self, json_file, json_dict):
        """
        write data dict into json file
        :return: True if sucess otherwise False
        :author: zzhou
        :date: 2018-06-06
        """
        json_data = json.dumps(json_dict)
        with open(json_file, 'w') as w:
            w.write(json_data)
        return True

    def get_standard_json(self):
        """
        the ai json result with default value for each key
        :return: the ai json result with default value for each key
        :author: zzhou
        :date: 2018-04-30
        """
        ai_dict = {}
        for key in ['patientID', 'studyUID', 'seriesUID', 'task', 'json_format_version']:
            ai_dict[key] = ''
        for key in ['slice_spacing', 'slice_thickness']:
            ai_dict[key] = None  # 0.0
        ai_dict['pixel_spacing'] = None  # (-0.0,-0.0)
        ai_dict['quality'] = None  # 0

        ai_dict['other_info'] = {}
        ai_dict['other_info']['ct_divol'] = None  # 0.0
        for key in ['kernel', 'manufacturer', 'patient_position', 'body_part']:
            ai_dict['other_info'][key] = None  # ''
        for key in ['kvp', 'current', 'number_slices']:
            ai_dict['other_info'][key] = None  # 0
        for key in ['lung_center', 'instance_range', 'image_size', 'image_position']:
            ai_dict['other_info'][key] = None  # (0,0)
        ai_dict['other_info']['nodule_align_lung_centor'] = None  # (0.0,0.0,0.0)

        ai_dict['nodes'] = None  # []

        return ai_dict

    def get_standard_node(self):
        """
        the ai node result with default value for each key
        :return: the ai node result with default value for each key
        :author: zzhou
        :date: 2018-04-30
        """
        node_dict = {}
        for key in ['node_index', 'note', 'type', 'GUID']:
            node_dict[key] = None  # ''
        node_dict['score'] = None  # 0.0
        node_dict['GUID'] = str(uuid.uuid1())  # unique id for node
        node_dict['label'] = None  # 1
        node_dict['confidence'] = None  # 0/1/2/3
        node_dict['from'] = None  # 0b000
        node_dict['attr'] = {}
        for key in ['density_type', 'position', 'mal_stage', 'obstructive_emphysema', 'tractivePleural',
                    'vascularnotch', 'spiculation', 'margin', 'necrosis', 'lobulation', 'cavitation',
                    'internal_structure', 'structure_type', 'mal_type', 'malignant', 'obstructive_pneumonia',
                    'mal_differentiation', 'calcification', 'air_bronchogram']:
            node_dict['attr'][key] = None  # ''

        node_dict['attr']['align'] = {}
        node_dict['attr']['align']['nodule_centor'] = None  # (0.0,0.0,0.0)
        node_dict['attr']['align']['radius_outer'] = None  # 0.0
        node_dict['attr']['align']['radius_inner'] = None  # 0.0

        node_dict['bounds'] = None  # []
        node_dict['rois'] = None  # []
        return node_dict

    # no use
    def get_standard_roi_bound(self):
        """
        the ai roi/bound result with default value for each key
        :return: the ai roi/bound result with default value for each key
        :author: zzhou
        :date: 2018-04-30
        """
        roi_dict = {}
        roi_dict['slice_index'] = None  # 0
        roi_dict['edge'] = None  # (0,0)
        return roi_dict

    def delete_dot_mark(self, json_file, is_overwrite=False):
        """
        replace '.,' with '.' in order to let json.load run without errors,
        for example, '123.,' -> '123,'
        :param json_file: json file full name
        :return: cover the original json file
        :author: zzhou
        :date: 2018-04-30
        """
        lines = None
        with open(json_file, 'r') as f:
            lines = f.readlines()
        assert type(lines) in [list], type(lines)
        new_lines = []
        for line in lines:
            new_line = line.replace('.,', ',')
            new_lines.append(new_line)

        if is_overwrite:
            with open(json_file, 'w') as f:
                f.writelines(new_lines)
        else:
            return new_lines

    def __assign_safe(self, info_dict, sub_dir, key, json_data=None, default=None):
        """
        get value from full_info safely (with default value)
        :return: value of the key
        :author: zzhou
        :date: 2018-04-30
        """
        if sub_dir in info_dict and key in info_dict[sub_dir]:
            return info_dict[sub_dir][key]
        if json_data is not None and key in json_data:
            return json_data[key]
        return default

    def convert_to_xyz_axis(self, point, pixel_spacing_3d, norm_rate=0.6):
        """
        convert normalized point to origin point
        :return: origin point
        :author:
        :date: 2018-08-20
        """
        if type(norm_rate) in [float, int]:
            norm_rate = [norm_rate] * 3
        if len(point) != 3 or not self.__is_number(point[0]):
            return None
        if type(pixel_spacing_3d) != np.ndarray or len(pixel_spacing_3d) != 3 \
                or not self.__is_number(pixel_spacing_3d[0]):
            return None

        dcm_axis = np.array(point) * norm_rate / pixel_spacing_3d
        return dcm_axis

    def __is_number(self, a):
        """
        judge a is a number or not
        :return: True or False
        :author:
        :date: 2018-08-20
        """
        try:
            x = float(a)
            return True
        except:
            return False

    def __round(self, a, p):
        """
        keep given number (p) float
        :param a: a given number
        :param p: precision
        :return: a number
        :author:
        :date: 2018-08-20
        """
        if self.__is_number(a) and self.__is_number(p):
            if p >= 0:
                return round(float(a), int(p))
            else:
                return float(a)
        else:
            return None

    def creat_bbox3d_from_detect_cube(self, x0, y0, z0, w, h, d, cut_float=-1):
        """
        return center points and size with given precision
        :param x0, y0, z0: start points of box
        :param w, h, d: bbox size
        :param cut_float: given precision
        :return: center points and size with given precision
        :author:
        :date: 2018-08-20
        """
        cx, cy, cz = x0 + (w - 1) * 0.5, y0 + (h - 1) * 0.5, z0 + (d - 1) * 0.5
        return [self.__round(cx, cut_float), self.__round(cy, cut_float), self.__round(cz, cut_float),
                self.__round(w, cut_float), self.__round(h, cut_float), self.__round(d, cut_float)]

    def __build_info_dict_from_dicom_dict(self, dicom_dict):
        """
        find sub_dir of dicom_dict and build info_dict from dicom_dict
        :param dicom_dict: series information
        :return: info_dict and sub_dir of dicom_dict
        :author:
        :date: 2018-08-20
        """
        json_data = {}
        json_data['seriesUID'] = dicom_dict['seriesUID']
        json_data['studyUID'] = dicom_dict['studyUID']
        json_data['patientID'] = dicom_dict['patientID']
        sub_dir = '/'.join([json_data['patientID'], json_data['studyUID'], json_data['seriesUID']])
        info_dict = {}
        info_dict[sub_dir] = dicom_dict
        return info_dict, sub_dir

    def make_json_meta_info(self, info_dict, sub_dir=None, json_format_version='2.0.0.180430',
                            taskName='LungNoduleDetect'):
        """
        generate meta info for json
        :param info_dict: series information
        :param sub_dir: sub_dir information of info_dict
        :return: json data for series
        :author:
        :date: 2018-08-20
        """
        if sub_dir is None:
            info_dict, sub_dir = self.__build_info_dict_from_dicom_dict(info_dict)
        json_class = StandardJSON()
        json_data = json_class.get_standard_json()
        json_data['patientID'], json_data['studyUID'], json_data['seriesUID'] = sub_dir.split('/')

        json_data['task'] = taskName
        json_data['json_format_version'] = json_format_version
        json_data['quality'] = 0
        json_data['slice_spacing'] = json_class.__assign_safe(info_dict, sub_dir, 'slice_spacing', None, None)
        json_data['slice_thickness'] = json_class.__assign_safe(info_dict, sub_dir, 'slice_thickness', None, None)
        json_data['pixel_spacing'] = json_class.__assign_safe(info_dict, sub_dir, 'pixel_spacing', None, None)

        json_data['other_info'] = {}
        json_data['other_info']['lung_center'] = None
        json_data['other_info']['kernel'] = json_class.__assign_safe(info_dict, sub_dir, 'kernel', default=None)
        json_data['other_info']['body_part'] = json_class.__assign_safe(info_dict, sub_dir, 'body_part', default=None)
        json_data['other_info']['image_position'] = json_class.__assign_safe(info_dict, sub_dir, 'image_position',
                                                                             default=None)
        json_data['other_info']['manufacturer'] = json_class.__assign_safe(info_dict, sub_dir, 'manufacturer',
                                                                           default=None)
        json_data['other_info']['ct_divol'] = json_class.__assign_safe(info_dict, sub_dir, 'ct_divol', default=None)
        json_data['other_info']['kvp'] = json_class.__assign_safe(info_dict, sub_dir, 'kvp', default=None)
        json_data['other_info']['current'] = json_class.__assign_safe(info_dict, sub_dir, 'current', default=None)
        json_data['other_info']['kvp'] = json_class.__assign_safe(info_dict, sub_dir, 'kvp', default=None)
        json_data['other_info']['instance_range'] = json_class.__assign_safe(info_dict, sub_dir, 'instance_range',
                                                                             default=None)
        json_data['other_info']['patient_position'] = json_class.__assign_safe(info_dict, sub_dir, 'patient_position',
                                                                               default=None)
        json_data['other_info']['number_slices'] = json_class.__assign_safe(info_dict, sub_dir, 'number_slices',
                                                                            default=None)
        json_data['other_info']['image_size'] = json_class.__assign_safe(info_dict, sub_dir, 'image_size', default=None)

        return json_data, json_class

    def make_json_result_from_bound3d(self, info_dict, bound3d_list, sub_dir=None, json_format_version='2.0.0.180430',
                                      taskName='LungNoduleDetect'):
        """
        save the json result of lung nodule detection from bound3d
        :param info_dict: series information
        :param bound3d_list: bound3d_list
        :param sub_dir: sub_dir information of info_dict
        :param json_format_version:json format version
        :param taskName:task name
        :return: json data to save
        :author:
        :date: 2018-08-20
        """
        if sub_dir is None:
            info_dict, sub_dir = self.__build_info_dict_from_dicom_dict(info_dict)

        json_data, json_class = self.make_json_meta_info(info_dict, sub_dir, json_format_version, taskName)

        instance_no_start = json_data['other_info']['instance_range'][0]

        pixel_spacing_3d = np.array(
            [json_data['pixel_spacing'][0], json_data['pixel_spacing'][1], json_data['slice_spacing']])
        json_data['nodes'] = []
        for node in bound3d_list:
            node_dict = json_class.get_standard_node()
            node_dict['from'] = 0b100
            node_dict['confidence'] = 0
            node_dict['score'] = float(node.score)
            node_dict['label'] = int(node.label)
            x, y, z, w, h, d = node.cube
            start_point = self.convert_to_xyz_axis((x, y, z), pixel_spacing_3d, norm_rate=0.6)
            end_point = self.convert_to_xyz_axis((x + w - 1, y + h - 1, z + d - 1), pixel_spacing_3d, norm_rate=0.6)

            node_dict['bounds'] = []
            for i in range(instance_no_start + int(start_point[2]),
                           instance_no_start + int(math.ceil(end_point[2])) + 1):
                edge = [
                    [float(start_point[0]), float(start_point[1])],
                    [float(end_point[0]), float(start_point[1])],
                    [float(end_point[0]), float(end_point[1])],
                    [float(start_point[0]), float(end_point[1])]
                ]
                node_dict['bounds'].append({'slice_index': i, 'edge': edge})
            json_data['nodes'].append(node_dict)

        return json_data

    def __check_borader(self, trans_z, trans_point, image_tensor_origin):
        """
        find the gap between seg pixel and background pixel
        :param trans_z: min or max value of z axis
        :param trans_point: seg points with trans_z axis
        :param image_tensor_origin: origin image tensor
        :return: gap_pixel
        :author:
        :date: 2018-08-16
        """
        trans = trans_z
        x_bound = image_tensor_origin.shape[2] - 1
        y_bound = image_tensor_origin.shape[1] - 1
        img_bound = numpy.empty((y_bound + 1, x_bound + 1), dtype='uint8')
        img_bound.fill(0)
        segpoint = trans_point
        segpoint = segpoint.astype(int)
        '''
        segpoint_copy = segpoint.reshape(segpoint.shape[0], 1, segpoint.shape[1])
        segpoint_list = []
        segpoint_list.append(segpoint_copy)
        '''
        cv2.drawContours(img_bound, (segpoint,), -1, color=255, thickness=-1)  # 轮廓和轮廓里面的区域都取值255
        x_start = min(segpoint[:, 0])
        x_end = max(segpoint[:, 0])
        y_start = min(segpoint[:, 1])
        y_end = max(segpoint[:, 1])
        interval = 2
        x_start = x_start - interval
        if x_start < 0:
            x_start = 0
        x_end = x_end + interval
        if x_end > x_bound:
            x_end = x_bound
        y_start = y_start - interval
        if y_start < 0:
            y_start = 0
        y_end = y_end + interval
        if y_end > y_bound:
            y_end = y_bound

        seg_pixel = []  #
        bound_pixel = []
        for i in range(0, y_end - y_start + 1):
            for j in range(0, x_end - x_start + 1):
                pixel_value = image_tensor_origin[trans, i + y_start, j + x_start]
                if img_bound[i + y_start][j + x_start] == 255:
                    seg_pixel.append(pixel_value)
                else:
                    bound_pixel.append(pixel_value)
        seg_pixel = numpy.array(seg_pixel)
        seg_pixel_median = numpy.median(seg_pixel)
        seg_pixel_median /= 255.
        bound_pixel = numpy.array(bound_pixel)
        bound_pixel_median = numpy.median(bound_pixel)
        bound_pixel_median /= 255.
        gap_pixel = seg_pixel_median - bound_pixel_median  # 找到seg点和非seg点在image tensor上中位数的差值
        return gap_pixel

    def __find_valid_rescale(self, center_x, center_y, origin_w, origin_h, img_width, img_height):
        """
        find a proper rescale for bounds to show
        :param center_x, center_y: x and y of center point
        :param origin_w, origin_h: w and h of bounds size
        :param img_width, img_height: w and h of origin image tensor
        :return: rescale
        :author: xqwang
        :date: 2018-08-20
        """
        return min(2 * center_x / origin_w, 2 * center_y / origin_h, 2 * (img_width - center_x) / origin_w,
                   2 * (img_height - center_y) / origin_h)

    def __get_rescale__bounds(self, center_point, bounds_size, img_shape, rescale):
        """
        find rescale_start_point and rescale_end_point for bounds to show
        :param center_point: center point of origin bounds
        :param bounds_size: size of origin bounds
        :param img_shape: origin image tensor shape
        :param rescale: rescale from path.conf
        :return: rescale_start_point and rescale_end_point
        :author: xqwang
        :date: 2018-08-20
        """
        center_x, center_y, center_z = center_point
        origin_w, origin_h, origin_d = bounds_size
        img_width, img_height = img_shape

        rescale_w, rescale_h = origin_w * rescale, origin_h * rescale
        rescale_start_point = numpy.array([center_x - rescale_w / 2, center_y - rescale_h / 2, center_z])
        rescale_end_point = rescale_start_point + numpy.array([rescale_w, rescale_h, 0])
        if rescale_start_point[0] >= 0 and rescale_start_point[1] >= 0 and rescale_end_point[0] <= img_width and \
                rescale_end_point[1] <= img_height:
            return rescale_start_point, rescale_end_point
        else:
            rescale = self.__find_valid_rescale(center_x, center_y, origin_w, origin_h, img_width, img_height)
            rescale_w, rescale_h = origin_w * rescale, origin_h * rescale
            rescale_start_point = numpy.array([center_x - rescale_w / 2, center_y - rescale_h / 2, center_z])
            rescale_end_point = rescale_start_point + numpy.array([rescale_w, rescale_h, 0])
            return rescale_start_point, rescale_end_point

    def __get_expand__bounds(self, center_point, bounds_size, img_shape, expand_pixel):
        """
        find rescale_start_point and rescale_end_point for bounds to show
        :param center_point: center point of origin bounds
        :param bounds_size: size of origin bounds
        :param img_shape: origin image tensor shape
        :param expand_pixel: rescale from path.conf
        :return: exapnd start point and expand end point
        :author: xqwang
        :date: 2018-08-27
        """
        center_x, center_y, center_z = center_point
        origin_w, origin_h, origin_d = bounds_size
        img_width, img_height = img_shape

        # -*- 如果结节小于  ，就固定到  -*-#
        # if max(bounds_size[:2]) < 10:
        #    rescale_w, rescale_h = 10, 10
        # else:
        rescale_w, rescale_h = origin_w + expand_pixel, origin_h + expand_pixel
        rescale_start_point = numpy.array([center_x - rescale_w / 2, center_y - rescale_h / 2, center_z])
        rescale_end_point = rescale_start_point + numpy.array([rescale_w, rescale_h, 0])
        if rescale_start_point[0] >= 0 and rescale_start_point[1] >= 0 and rescale_end_point[0] <= img_width and \
                rescale_end_point[1] <= img_height:
            return rescale_start_point, rescale_end_point
        else:
            rescale = self.__find_valid_rescale(center_x, center_y, origin_w, origin_h, img_width, img_height)
            rescale_w, rescale_h = origin_w * rescale, origin_h * rescale
            rescale_start_point = numpy.array([center_x - rescale_w / 2, center_y - rescale_h / 2, center_z])
            rescale_end_point = rescale_start_point + numpy.array([rescale_w, rescale_h, 0])
            return rescale_start_point, rescale_end_point

    def __get_new_expand__bounds_list(self, old_bounds_list, img_shape, expand_pixel, cut_float):
        """
        find new nodule bounds list by expand some pixel
        :param old_bounds_list: old_bounds_list without expand
        :param img_shape: origin image tensor shape
        :param expand_pixel: expand pixel path.conf
        :return: exapnd new bounds list
        """
        new_bounds_list = []
        img_width, img_height = img_shape
        for old_bound_dict in old_bounds_list:
            slice_z = old_bound_dict['slice_index']
            old_edge = old_bound_dict['edge']
            old_edge_np = np.array(old_edge)
            old_x_min, old_y_min = np.min(old_edge_np, axis=0)
            old_x_max, old_y_max = np.max(old_edge_np, axis=0)
            new_x_min = old_x_min - expand_pixel / 2
            new_y_min = old_y_min - expand_pixel / 2
            if new_x_min < 0:
                new_x_min = 0.
            if new_y_min < 0:
                new_y_min = 0.
            new_x_max = old_x_max + expand_pixel / 2
            new_y_max = old_y_max + expand_pixel / 2
            if new_x_max >= img_width:
                new_x_max = img_width - 1.
            if new_y_max >= img_height:
                new_y_max = img_height - 1.
            # edge = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            new_edge = [
                [self.__round(new_x_min, cut_float), self.__round(new_y_min, cut_float)],
                [self.__round(new_x_max, cut_float), self.__round(new_y_min, cut_float)],
                [self.__round(new_x_max, cut_float), self.__round(new_y_max, cut_float)],
                [self.__round(new_x_min, cut_float), self.__round(new_y_max, cut_float)]
            ]
            new_bounds_list.append({'slice_index': slice_z, 'edge': new_edge})
        return new_bounds_list

    def __get_expand__roi_bounds(self, points_list, instance_dict_no, img_shape, nodule_bounds_expand_pixel):
        points_array = numpy.array(points_list)
        points_array_x, points_array_y = points_array[:, 0], points_array[:, 1]
        x_min, x_max = numpy.min(points_array_x), numpy.max(points_array_x)
        y_min, y_max = numpy.min(points_array_y), numpy.max(points_array_y)
        r_w, r_h = x_max - x_min, y_max - y_min
        center_point = (x_min + r_w / 2, y_min + r_h / 2, instance_dict_no)
        rescale_start_point, rescale_end_point = self.__get_expand__bounds(center_point, numpy.array([r_w, r_h, -1]),
                                                                           img_shape, nodule_bounds_expand_pixel)
        rescale_start_point[2], rescale_end_point[2] = instance_dict_no, instance_dict_no

        edge = [[rescale_start_point[0], rescale_start_point[1]], [rescale_end_point[0], rescale_start_point[1]],
                [rescale_end_point[0], rescale_end_point[1]], [rescale_start_point[0], rescale_end_point[1]]]

        return edge

    def __points_to_nodule_dict(self, one_seg_points_all, mal_prob, series, instance_dict, cut_float,
                                nodule_bounds_expand_pixel, ai_type='thin'):  # 每一个bound_seg
        """
        save the json result of every lung nodule
        :param one_seg_points_all: seg points information in one bound
        :param series: series information
        :param instance_dict: series instance_no dict
        :param cut_float: precision of float number
        :param nodule_bounds_rescale: rescale for nodule bounds to show
        :return: a nodule data
        :author:
        :date: 2018-08-16
        """
        try:
            one_seg_points = []
            c_x, c_y, c_z, c_w, c_h, c_d = one_seg_points_all[
                5]  # 取的是bound_seg['coord_raw'] (5, 0.6, 0.6) or (0.6, 0.6, 0.6)
            # c_x, c_y, c_z, c_w, c_h, c_d = one_seg_points_all[1]  # 取的是bound_seg['coord']
            if ai_type == 'thick':
                norm_rate = (0.6, 0.6, 5.0)
                start_point = (c_x, c_y, c_z)
                end_point = (c_x + c_w - 1, c_y + c_h - 1, c_z + c_d - 1)
                center_point = (c_x + (c_w - 1) / 2, c_y + (c_h - 1) / 2, c_z + (c_d - 1) / 2)
            else:
                norm_rate = (0.6, 0.6, 0.6)
                start_point = (c_x, c_y, c_z)
                end_point = (c_x + c_w, c_y + c_h, c_z + c_d)
                center_point = (c_x + c_w / 2, c_y + c_h / 2, c_z + c_d / 2)
            bounds_size = (c_w, c_h, c_d)
            pixel_spacing_3d = numpy.array([float(series['pixel_spacing'][0]), float(series['pixel_spacing'][1]),
                                            float(series['slice_spacing'])])
            start_point = numpy.array(start_point) * numpy.array(norm_rate) / pixel_spacing_3d
            end_point = numpy.array(end_point) * numpy.array(norm_rate) / pixel_spacing_3d  # 还原到原来的image坐标
            center_point = numpy.array(center_point) * numpy.array(norm_rate) / pixel_spacing_3d
            bounds_size = numpy.array(bounds_size) * numpy.array(norm_rate) / pixel_spacing_3d

            origin_one_seg_points = one_seg_points_all[7]
            int_slice_dict = {}
            if len(origin_one_seg_points) > 0:
                for origin_point in origin_one_seg_points:  # 每一个contour
                    trans_point = numpy.array(origin_point)
                    one_seg_points.append(trans_point)
                slice_dict = {}
                for point in one_seg_points:
                    point_x, point_y, point_z = point[:3]
                    point_z = int(round(point_z))
                    if point_z >= max(instance_dict.keys()):
                        continue

                    if point_z not in slice_dict.keys():
                        slice_dict[point_z] = []
                    slice_dict[point_z].append([point_x, point_y, point[3]])  # x,y,很多个点，point[3]是contour_index

                # added to generate proper results
                z_axis = slice_dict.keys()
                if len(z_axis) <= 0:
                    print("[Waring] slice_dict is 0")
                    return None
                z_axis_min = int(min(z_axis))
                z_axis_max = int(math.ceil(max(z_axis)))

                for z in range(z_axis_min, z_axis_max + 1):  # 将离z最近的slice_dict的point对应到z上
                    abs_delta = abs(numpy.array(z_axis) - z).tolist()
                    idx = abs_delta.index(min(abs_delta))  # 只会找1个
                    int_slice_dict[z] = slice_dict[z_axis[idx]]  # 很多个[point[0], point[1]]
                    '''
                    if z_axis_min < z_axis_max: # 判断要不要删除int_slice_dict[z_axis_max]或者int_slice_dict[z_axis_min]
                    image_tensor_origin = series['img_tensor']
                    gap_min = self.__check_borader(z_axis_min, numpy.array(int_slice_dict[z_axis_min])[:,:2],
                                                   image_tensor_origin)
                    gap_max = self.__check_borader(z_axis_max, numpy.array(int_slice_dict[z_axis_max])[:,:2],
                                                   image_tensor_origin)
                    z_min_del = False
                    z_max_del = False
                    if gap_min < 0.15:
                        z_min_del = True
                    if gap_max < 0.15:
                        z_max_del = True
                    if (z_axis_max - z_axis_min == 1 and z_max_del and z_min_del):
                        if gap_min > gap_max:
                            z_min_del = False
                        else:
                            z_max_del = False
                    # botong do CRF do not del slice 4 lines
                    if z_max_del:
                        del int_slice_dict[z_axis_max]
                    if z_min_del:
                        del int_slice_dict[z_axis_min]
                    '''

            nodule = {}
            nodule['bbox3d'] = []
            x0, y0, z0 = start_point
            x1, y1, z1 = end_point
            w, h, d = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
            bbox3d = self.creat_bbox3d_from_detect_cube(x0, y0, z0, w, h, d, cut_float)  # 转换成中心点和边长，保留两位小数
            nodule['bbox3d'] = bbox3d

            # nodule['slice_number'] = len(int_slice_dict.keys())
            nodule['bounds'] = []
            nodule['bounds_AI'] = []
            #########Definite start and stop layer##############
            min_key = min(instance_dict.keys())
            max_key = max(instance_dict.keys())
            start = int(round(start_point[2]))
            start = start if start >= min_key else min_key
            stop = int(round(end_point[2]))
            stop = stop if stop <= max_key else max_key

            start_instance_no = instance_dict[start]
            stop_instance_no = instance_dict[stop]
            if ai_type == 'thick':
                stop_instance_no += 1
            else:
                if stop_instance_no == start_instance_no:
                    if stop_instance_no + 1 <= instance_dict[max_key]:
                        stop_instance_no += 1
                    elif start_instance_no - 1 >= instance_dict[min_key]:
                        start_instance_no -= 1
            remove_seg_z = True
            if remove_seg_z:
                seg_z_list = int_slice_dict.keys()
                for seg_z in seg_z_list:
                    if instance_dict[seg_z] not in range(start_instance_no, stop_instance_no):
                        del int_slice_dict[seg_z]

            ###############Definite start and end layer##############
            # rescale_start_point, rescale_end_point = self.__get_expand__bounds(center_point, bounds_size, series['image_size'], nodule_bounds_expand_pixel)
            # rescale_start_point[2], rescale_end_point[2] = start_point[2], end_point[2]
            rescale_start_point, rescale_end_point = start_point, end_point

            instance_no_idx_dict = {}
            idx = 0
            for i in range(start_instance_no, stop_instance_no):
                edge = [
                    [self.__round(rescale_start_point[0], cut_float), self.__round(rescale_start_point[1], cut_float)],
                    [self.__round(rescale_end_point[0], cut_float), self.__round(rescale_start_point[1], cut_float)],
                    [self.__round(rescale_end_point[0], cut_float), self.__round(rescale_end_point[1], cut_float)],
                    [self.__round(rescale_start_point[0], cut_float), self.__round(rescale_end_point[1], cut_float)]
                ]

                nodule['bounds'].append({'slice_index': i, 'edge': edge})  # 用于界面呈现
                instance_no_idx_dict[i] = idx  # instance_no和对应的在nodule['bounds']的index

                edge = [
                    [self.__round(start_point[0], cut_float), self.__round(start_point[1], cut_float)],
                    [self.__round(end_point[0], cut_float), self.__round(start_point[1], cut_float)],
                    [self.__round(end_point[0], cut_float), self.__round(end_point[1], cut_float)],
                    [self.__round(start_point[0], cut_float), self.__round(end_point[1], cut_float)]
                ]
                nodule['bounds_AI'].append({'slice_index': i, 'edge': edge})  # 原始检测框

                idx = idx + 1

            nodule['rois'] = []
            if len(origin_one_seg_points) > 0:
                for key, value in int_slice_dict.items():  # 每一个slice的多条轮廓的多个轮廓点，由此得到roi、长径和短径
                    if len(value) < 3:  # 少于3个点的轮廓直接跳过
                        continue
                    points_list = []
                    if len(value) < 2:
                        non_repeat_points = np.float32(value)[:, :2].tolist()
                        nodule['rois'].append({'slice_index': instance_dict[key], 'edge': non_repeat_points})
                        points_list = non_repeat_points
                    else:
                        contour_index = np.array(value)[:, 2]
                        contour_unique = np.unique(contour_index).tolist()
                        contour_num = len(contour_unique)
                        if contour_num == 1:
                            non_repeat_points = []
                            point_xy = np.float32(value)[:, :2].tolist()
                            last_x, last_y = point_xy[-1]
                            for raw_x, raw_y in point_xy:
                                if raw_x == last_x and raw_y == last_y:
                                    continue
                                non_repeat_points.append(
                                    [self.__round(raw_x, cut_float), self.__round(raw_y, cut_float)])
                                last_x, last_y = raw_x, raw_y
                            if len(non_repeat_points) != 0:
                                nodule['rois'].append({'slice_index': instance_dict[key], 'edge': non_repeat_points})
                                points_list = non_repeat_points
                        else:  # 属于不同的contour index的point要存成slice_index相同的多个rois
                            for c_idx in contour_unique:
                                idx = contour_index == c_idx
                                non_repeat_points = []
                                point_xy = np.float32(value)[idx, :2].tolist()
                                last_x, last_y = point_xy[-1]
                                for raw_x, raw_y in point_xy:
                                    if raw_x == last_x and raw_y == last_y:
                                        continue
                                    non_repeat_points.append(
                                        [self.__round(raw_x, cut_float), self.__round(raw_y, cut_float)])
                                    points_list.append([self.__round(raw_x, cut_float), self.__round(raw_y, cut_float)])
                                    last_x, last_y = raw_x, raw_y
                                if len(non_repeat_points) != 0:
                                    nodule['rois'].append(
                                        {'slice_index': instance_dict[key], 'edge': non_repeat_points})

                    if len(points_list) != 0:  # 同一个结节在一张slice上的1个或多个roi包含在1个展示框内
                        edge = self.__get_expand__roi_bounds(points_list, instance_dict[key], series['image_size'],
                                                             nodule_bounds_expand_pixel=0)
                        if instance_dict[key] in instance_no_idx_dict.keys():  # 如果该instance_no的'bounds'已经存在，则取并集替换
                            x_min = min(
                                min(n[0] for n in nodule['bounds'][instance_no_idx_dict[instance_dict[key]]]['edge']),
                                min(e[0] for e in edge))
                            x_max = max(
                                max(n[0] for n in nodule['bounds'][instance_no_idx_dict[instance_dict[key]]]['edge']),
                                max(e[0] for e in edge))
                            y_min = min(
                                min(n[1] for n in nodule['bounds'][instance_no_idx_dict[instance_dict[key]]]['edge']),
                                min(e[1] for e in edge))
                            y_max = max(
                                max(n[1] for n in nodule['bounds'][instance_no_idx_dict[instance_dict[key]]]['edge']),
                                max(e[1] for e in edge))
                            edge = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                            nodule['bounds'][instance_no_idx_dict[instance_dict[key]]]['edge'] = edge  #
                        else:  # 否则则添加
                            nodule['bounds'].append({'slice_index': instance_dict[key], 'edge': edge})

            expand_bounds_list = self.__get_new_expand__bounds_list(nodule['bounds'], series['image_size'],
                                                                    nodule_bounds_expand_pixel, cut_float)
            nodule['bounds'] = expand_bounds_list
            nodule['attr'] = {'density_type': '', 'position': '', 'location': ''}
            if len(one_seg_points_all) > 2:
                nodule['label'] = str(one_seg_points_all[2])
                if str(one_seg_points_all[2]) in ['3', '4']:
                    nodule['attr']['density_type'] = 'ggo'
                elif str(one_seg_points_all[2]) in ['1', '2']:
                    nodule['attr']['density_type'] = 'solid'
                elif str(one_seg_points_all[2]) in ['5', '6']:
                    nodule['attr']['density_type'] = 'cal'
                ###linhaoliang1026###
                elif str(one_seg_points_all[2]) in ['7', '8']:
                    nodule['attr']['density_type'] = 'part'
                ###linhaoliang1026###
            if len(one_seg_points_all) > 3:
                nodule['score'] = float(str(one_seg_points_all[3]))

            if mal_prob is not None:
                nodule['attr']['malignant'] = float(mal_prob)
            return nodule
        except Exception as ex:
            print("[Exception] in __points_to_nodule_dict")
            import traceback
            traceback.print_exc()
            return None

    def make_json_result_from_seg_points(self, dicom_dict, seg_points, mal_probs, attr_results, instance_dict,
                                         loc_results, nodule_bounds_expand_pixel, lung_center,
                                         nodule_align_lung_center, nodule_align_info_list,
                                         json_format_version='2.0.0.180430', taskName='LungNoduleDetect', cut_float=2,
                                         ai_type='thin', display=False):
        """
        save the json result of lung nodule detection from seg_points
        :param dicom_dict: series information
        :param seg_points: seg points information
        :param attr_results: attribute prediction
        :param instance_dict: series instance_no dict
        :param lobePos: nodule location information
        :param nodule_bounds_rescale: rescale for nodule bounds to show
        :param json_format_version:json format version
        :param taskName:task name
        :param cut_float: precision of float number
        :return: json data to save
        :author:
        :date: 2018-08-16
        """
        json_data, json_class = self.make_json_meta_info(dicom_dict, None, json_format_version=json_format_version,
                                                         taskName=taskName)
        json_data['other_info']['lung_center'] = lung_center
        json_data['other_info']['multi_nodule'] = display
        json_data['other_info']['nodule_align_lung_centor'] = nodule_align_lung_center

        if len(loc_results) < len(seg_points):
            print('less lobePos', len(loc_results), 'seg_points', len(seg_points))
            keypoint = namedtuple("keypoint", ["index", "pos", "lobe", "segment"])
            loc_results = [keypoint(None, None, "", "") for _ in range(len(seg_points))]
        nodule_list = []
        # nodule_loc_list = []
        # print "lobePos", loc_results

        if seg_points is not None:
            for index, one_seg_points in enumerate(seg_points):
                # if index == 2:
                #    pdb.set_trace()
                if one_seg_points == []:
                    print('Exception one_seg_points is None')
                    continue
                if mal_probs is not None:
                    nodule = self.__points_to_nodule_dict(one_seg_points, mal_probs[index], dicom_dict, instance_dict,
                                                          cut_float, nodule_bounds_expand_pixel, ai_type=ai_type)
                else:
                    nodule = self.__points_to_nodule_dict(one_seg_points, None, dicom_dict, instance_dict, cut_float,
                                                          nodule_bounds_expand_pixel, ai_type=ai_type)
                if nodule is None:
                    print("[Waring] nodule is None")
                    continue
                nodule['GUID'] = str(uuid.uuid1())
                nodule['node_index'] = index
                nodule['type'] = 'Lung_Nodule'
                nodule['from'] = 0b100
                nodule['confidence'] = 100
                nodule['note'] = ''
                nodule['attr']['location'] = loc_results[index].lobe
                nodule['attr']['nodule_segment'] = loc_results[index].lobe + loc_results[index].segment
                if len(nodule_align_info_list) > 0:
                    nodule['attr']['nodule_align'] = nodule_align_info_list[index]
                # botong 190219
                if attr_results is not None:
                    for key in attr_results[index]:
                        nodule['attr'][key] = attr_results[index][key]

                nodule_list.append(nodule)
                # print nodule['attr']['location']
                # print nodule['attr']['nodule_segment']
                # print nodule['attr']['nodule_align']

            json_data['nodes'] = nodule_list
        return json_data

    def make_json_result_from_detect(self, dicom_dict, detect_result, json_format_version='2.0.0.180430',
                                     taskName='LungNoduleDetect', cut_float=2, norm_spacing=(0.6, 0.6, 0.6),
                                     display=False):
        """
        save the json result of lung nodule detection from detect_result
        :param dicom_dict: series information
        :param detect_result: seg points information
        :param instance_dict: series instance_no dict
        :param lobePos: nodule location information
        :param nodule_bounds_rescale: rescale for nodule bounds to show
        :param json_format_version:json format version
        :param taskName:task name
        :param cut_float: precision of float number
        :return: json data to save
        :author:
        :date: 2018-08-16
        """
        json_data, json_class = self.make_json_meta_info(dicom_dict, None, json_format_version=json_format_version,
                                                         taskName=taskName)

        instance_no_start = json_data['other_info']['instance_range'][0]
        json_data['other_info']['multi_nodule'] = display

        pixel_spacing_3d = np.array(
            [json_data['pixel_spacing'][0], json_data['pixel_spacing'][1], json_data['slice_spacing']])
        json_data['nodes'] = []
        for node_index, node in enumerate(detect_result):
            # node: x, y, z, w, h, d, t, label, score
            node_dict = json_class.get_standard_node()
            node_dict['from'] = 0b100
            node_dict['confidence'] = 0
            node_dict['score'] = float(node[8])
            node_dict['label'] = int(node[7])
            node_dict['GUID'] = str(uuid.uuid1())
            node_dict['node_index'] = node_index
            node_dict['type'] = 'Lung_Nodule'
            x, y, z, w, h, d = node[:6]
            start_point = self.convert_to_xyz_axis((x, y, z), pixel_spacing_3d, norm_rate=norm_spacing)
            end_point = self.convert_to_xyz_axis((x + w - 1, y + h - 1, z + d - 1), pixel_spacing_3d,
                                                 norm_rate=norm_spacing)
            if int(start_point[2]) == int(math.ceil(end_point[2])):
                end_point[2] += 1

            node_dict['bounds'] = []
            node_dict['rois'] = []
            # for i in xrange(instance_no_start + int(start_point[2]), instance_no_start + int(math.ceil(end_point[2])) + 1):
            for i in range(instance_no_start + int(start_point[2]), instance_no_start + int(math.ceil(end_point[2]))):
                edge = [
                    [self.__round(start_point[0], cut_float), self.__round(start_point[1], cut_float)],
                    [self.__round(end_point[0], cut_float), self.__round(start_point[1], cut_float)],
                    [self.__round(end_point[0], cut_float), self.__round(end_point[1], cut_float)],
                    [self.__round(start_point[0], cut_float), self.__round(end_point[1], cut_float)]
                ]
                node_dict['bounds'].append({'slice_index': i, 'edge': edge})
            json_data['nodes'].append(node_dict)

        return json_data

    def make_json_result_from_detect_not_norm(self, dicom_dict, detect_result, json_format_version='2.0.0.180430',
                                              taskName='LungNoduleDetect', cut_float=2, norm_spacing=(0.6, 0.6, 0.6),
                                              display=False):
        """
        save the json result of lung nodule detection from detect_result
        :param dicom_dict: series information
        :param detect_result: seg points information
        :param instance_dict: series instance_no dict
        :param lobePos: nodule location information
        :param nodule_bounds_rescale: rescale for nodule bounds to show
        :param json_format_version:json format version
        :param taskName:task name
        :param cut_float: precision of float number
        :return: json data to save
        :author:
        :date: 2018-08-16
        """
        json_data, json_class = self.make_json_meta_info(dicom_dict, None, json_format_version=json_format_version,
                                                         taskName=taskName)
        json_data['other_info']['multi_nodule'] = display
        instance_no_start = json_data['other_info']['instance_range'][0]

        json_data['nodes'] = []
        for node_index, node in enumerate(detect_result):
            # node: x, y, z, w, h, d, t, label, score
            bounds, score = node
            new_bounds = []
            for bound in bounds:
                edge = [[self.__round(x, cut_float), self.__round(y, cut_float)] for (x, y) in bound['edge']]
                slice_index = bound['slice_index'] + instance_no_start
                new_bounds.append({'edge': edge, 'slice_index': slice_index})
            node_dict = json_class.get_standard_node()
            node_dict['from'] = 0b100
            node_dict['confidence'] = 0
            node_dict['score'] = float(score)
            node_dict['label'] = 1
            node_dict['GUID'] = str(uuid.uuid1())
            node_dict['node_index'] = node_index
            node_dict['type'] = 'Lung_Nodule'

            node_dict['bounds'] = new_bounds
            node_dict['rois'] = []
            json_data['nodes'].append(node_dict)

        return json_data


# ---------------------------------------------------------------------------------------------
# MERGE 2D BOX TO 3D
# ---------------------------------------------------------------------------------------------
class UnionFind(object):
    """
    union find class
    example:

    """

    def __init__(self, groups):
        """
        :param groups: list of tuples, each tuple means a set.
            Elements in tuples can be numbers or strings.
        """
        self.groups = groups
        items = []
        for g in groups:
            items += g
        self.items = set(items)
        self.parent = {}
        self.tree_size_by_root = {}  # indexed by root
        for item in self.items:
            self.tree_size_by_root[item] = 1
            self.parent[item] = item

    def union(self, item1, item2):
        r1 = self.find_root(item1)
        r2 = self.find_root(item2)
        if r1 == r2:
            return
        size1 = self.tree_size_by_root[r1]
        size2 = self.tree_size_by_root[r2]
        if size1 >= size2:
            min_tree_root, max_tree_root = r2, r1
        else:
            min_tree_root, max_tree_root = r1, r2
        self.parent[min_tree_root] = max_tree_root
        self.tree_size_by_root[max_tree_root] += self.tree_size_by_root[min_tree_root]
        self.tree_size_by_root.pop(min_tree_root)

    def find_root(self, r):
        if r in self.tree_size_by_root.keys():
            return r
        else:
            return self.find_root(self.parent[r])

    def run(self):
        res_groups = []
        for g in self.groups:
            g_size = len(g)
            if g_size >= 2:
                for i in range(1, g_size):
                    self.union(g[i - 1], g[i])
        groups_by_root = {}
        for item in self.items:
            root = self.find_root(item)
            groups_by_root.setdefault(root, [])
            groups_by_root[root].append(item)
        for root, group in groups_by_root.items():
            res_groups.append(group)
        return res_groups


def round_int(x):
    return int(np.round(x))


def check_rect(rect):
    """
    Check valid Rectangle
    """
    x, y, w, h = rect
    return x >= 0 and y >= 0 and w > 0 and h > 0


def rect_intersect(r1, r2):
    """
    Intersect two rectangle
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if not check_rect((x1, y1, w1, h1)):
        return -1, -1, -1, -1
    if not check_rect((x2, y2, w2, h2)):
        return -1, -1, -1, -1
    x1_ = x1 + w1
    y1_ = y1 + h1
    x2_ = x2 + w2
    y2_ = y2 + h2
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    x3_ = min(x1_, x2_)
    y3_ = min(y1_, y2_)
    if x3_ <= x3 or y3_ <= y3:
        return -1, -1, -1, -1
    return x3, y3, x3_ - x3, y3_ - y3


def boxlist_to_arr12(box_list, roi_list):

    # box_list is (x1,y1,x2,y2,slice_idx,label,score).
    roi_ct = [item for sublist in roi_list for item in sublist]

    # ATTENTION: concatenate remove the image has no box
    results_i = np.concatenate(box_list, axis=0)
    z_dimension = results_i.shape[0]
    res_arr12 = np.zeros(shape=(z_dimension, 12)).astype(results_i.dtype)

    # x_top, y_top, z_top, w, h, d, 1, label, score
    res_arr12[:, :2] = results_i[:, :2]
    res_arr12[:, 2] = results_i[:, 4]
    res_arr12[:, 3] = results_i[:, 2] - results_i[:, 0]
    res_arr12[:, 4] = results_i[:, 3] - results_i[:, 1]
    res_arr12[:, 5] = 1
    res_arr12[:, 7] = results_i[:, 5]
    res_arr12[:, 8] = results_i[:, 6]

    # sort by z
    inds = res_arr12[:, 2].argsort()
    res_arr12 = res_arr12[inds]
    roi_array = np.array(roi_ct)
    roi_reorder = roi_array[inds]

    return res_arr12, roi_reorder


# ---------------------------------------------------------------------------------------------
# merge 2d to 3d
# ---------------------------------------------------------------------------------------------
def merge_bounds_3d(dets_ct, rois_ct, merge_thresh, cls_num, max_slices_stride=1,
                    iom_thresh=0.7, use_max=False, use_moving_avg=True):

    combine_opt = {}
    # per-class-threshold is defined
    if isinstance(merge_thresh, list):
        thresh_list = merge_thresh
    else:
        thresh_list = [merge_thresh for _ in range(cls_num)]

    label_combine_matrix = np.ones((cls_num, cls_num), dtype='int32') * -1
    label_combine_matrix[range(1, cls_num), range(1, cls_num)] = range(1, cls_num)
    least_inter_ratio_matrix = np.zeros((cls_num, cls_num), dtype='float32')
    least_inter_ratio_matrix[range(1, cls_num), range(1, cls_num)] = \
        np.ones((cls_num-1), dtype='float32') * iom_thresh

    combine_opt['label_combine_matrix'] = label_combine_matrix
    combine_opt['least_inter_ratio_matrix'] = least_inter_ratio_matrix
    combine_opt['max_slices_stride'] = max_slices_stride

    to_merge = []
    to_merge_roi = []
    for idx in range(dets_ct.shape[0]):
        slice_det = dets_ct[idx]
        slice_det = slice_det.tolist()
        x, y, z, w, h, d = slice_det[:6]
        label = int(slice_det[7])
        score = slice_det[8]
        if score > thresh_list[label]:
            to_merge.append([x, y, w, h, int(z), label, score])
            to_merge_roi.append(rois_ct[idx])

    bound2d_group = _combine_bounds_3d_direct(to_merge, to_merge_roi, combine_opt)
    merged_bound3d = _bound_group_to_bound2ds(bound2d_group, use_max, use_moving_avg)
    return merged_bound3d


def merge_bounds_3d_ori(dets_ct, rois_ct, merge_thresh, cls_num, max_slices_stride=1,
                    iom_thresh=0.7, use_max=False, use_moving_avg=True):

    combine_opt = {}
    # per-class-threshold is defined
    if isinstance(merge_thresh, list):
        thresh_list = merge_thresh
    else:
        thresh_list = [merge_thresh for _ in range(cls_num)]

    label_combine_matrix = np.ones((cls_num, cls_num), dtype='int32') * -1
    label_combine_matrix[range(1, cls_num), range(1, cls_num)] = range(1, cls_num)
    least_inter_ratio_matrix = np.zeros((cls_num, cls_num), dtype='float32')
    least_inter_ratio_matrix[range(1, cls_num), range(1, cls_num)] = \
        np.ones((cls_num-1), dtype='float32') * iom_thresh

    combine_opt['label_combine_matrix'] = label_combine_matrix
    combine_opt['least_inter_ratio_matrix'] = least_inter_ratio_matrix
    combine_opt['max_slices_stride'] = max_slices_stride

    to_merge = []
    to_merge_roi = []
    for idx in range(dets_ct.shape[0]):
        slice_det = dets_ct[idx]
        slice_det = slice_det.tolist()
        x, y, z, w, h, d = slice_det[:6]
        label = int(slice_det[7])
        score = slice_det[8]
        if score > thresh_list[label]:
            to_merge.append([x, y, w, h, int(z), label, score])
            to_merge_roi.append(rois_ct[idx])

    bound2d_group = _combine_bounds_3d_direct(to_merge, to_merge_roi, combine_opt)
    return bound2d_group


def _combine_bounds_3d_direct(bound2ds_list, rois_ct, opt):

    label_combine_matrix = opt['label_combine_matrix']
    least_inter_ratio_matrix = opt['least_inter_ratio_matrix']
    max_slices_stride = opt['max_slices_stride']

    num_bounds = len(bound2ds_list)
    num_slice = 0
    for bound2d in bound2ds_list:
        num_slice = max(num_slice, bound2d[4])
    num_slice += 1

    # combine all bounds through union-find
    # 1) list all bounds so that each bound can have an id
    bound2ds_ids_by_slice = [[] for i in range(num_slice)]
    for (idx, bound2d) in enumerate(bound2ds_list):
        slice_id = bound2d[4]
        bound2ds_ids_by_slice[slice_id].append({'id': idx, 'bound': bound2d})

    # 2) find all pairs of bounds that can be combined
    combine_pairs = []
    for i in range(num_bounds):
        combine_pairs.append((i,))
    for slice_id1 in range(num_slice):
        # find pairs inside current slice and current-slice-vs-next-max_slices_stride-slices
        slice_bounds_with_ids1 = bound2ds_ids_by_slice[slice_id1]
        slice_bounds_num1 = len(slice_bounds_with_ids1)
        for slice_id2 in range(slice_id1, min(num_slice, slice_id1 + max_slices_stride + 1)):
            slice_bounds_with_ids2 = bound2ds_ids_by_slice[slice_id2]
            slice_bounds_num2 = len(slice_bounds_with_ids2)
            for i in range(slice_bounds_num1):
                if slice_id1 == slice_id2:
                    j_start = i + 1
                else:
                    j_start = 0
                for j in range(j_start, slice_bounds_num2):
                    b1 = slice_bounds_with_ids1[i]['bound']
                    b2 = slice_bounds_with_ids2[j]['bound']
                    id1 = slice_bounds_with_ids1[i]['id']
                    id2 = slice_bounds_with_ids2[j]['id']
                    if _check_merge_pairwise(b1, b2, label_combine_matrix, least_inter_ratio_matrix):
                        combine_pairs.append((id1, id2))

    # 3) union find
    uf = UnionFind(combine_pairs)
    combined_bound_ids = uf.run()

    # 4) combine the bounds
    bound_groups = []
    for bound_ids in combined_bound_ids:
        bound_group = []
        for b_id in bound_ids:
            bound_group.append([bound2ds_list[b_id], rois_ct[b_id]])
        bound_groups.append(bound_group)

    return bound_groups


_key2index = {
    "x": 0,
    "y": 1,
    "w": 2,
    "h": 3,
    "z": 4,
    "label": 5,
    "score": 6
}


def _bound_group_to_bound2ds(bound_groups, use_max=False, use_moving_avg=True):
    """
        Convert groups of bound2d to a list of {'slice_index': xx, 'edge': (x1, y1, x2, y2)}
        input params: Merged bound3ds, i.e. a list of bound3d
        output: Groups of [bound2d, roi2d, avg_score, label] list.
    """
    merged_bound3d = []
    for bound_group in bound_groups:
        bound2ds = []
        roi2ds = []
        avg_dict = {}
        label = bound_group[0][0][_key2index['label']]
        for bound in bound_group:
            slice_id = bound[0][_key2index['z']]
            x1, y1 = bound[0][_key2index['x']], bound[0][_key2index['y']]
            x2 = x1 + max(0, bound[0][_key2index['w']] - 1)
            y2 = y1 + max(0, bound[0][_key2index['h']] - 1)
            roi = bound[1]
            # for each slice, only the max score are kept for score fusion to yield the 3D score.
            if slice_id not in avg_dict:
                avg_dict[slice_id] = bound[0][_key2index['score']]
            else:
                avg_dict[slice_id] = max(bound[0][_key2index['score']], avg_dict[slice_id])
            bound2ds.append({'slice_index': slice_id, 'edge': (x1, y1, x2, y2)})
            roi2ds.append({'slice_index': slice_id, 'edge': roi})

        bound3d_score = _get_score(avg_dict, use_max, use_moving_avg)
        merged_bound3d.append([bound2ds, roi2ds, bound3d_score, label])
    return merged_bound3d


def _get_score(avg_dict, use_max=False, use_moving_avg=True):
    if use_moving_avg:
        avg_dict = _get_moving_dict(avg_dict)
    return _calculate_score(avg_dict, use_max)


def get_score_one(bound_groups, moving_avg=False):
    """Get score for all bound groups"""
    # [x, y, w, h, int(z), label, score]
    if True:
        score_count = 0
        avg_dict = {}
        for bound in bound_groups:
            avg_dict[bound[0][4]] = bound[0][6]
            score_count += bound[0][6]
        if moving_avg:
            moving_avg_score, moving_max_score = get_moving_avg(avg_dict)
            score = moving_avg_score
        else:
            score = score_count/len(bound_groups)
    return score


def _calculate_score(avg_dict, use_max=False):
    total_score_list = [score for score in avg_dict.values()]
    if use_max:
        return np.max(total_score_list)
    else:
        return np.mean(total_score_list)


def _get_moving_dict(avg_dict):
    slice_list = list(avg_dict.keys())
    slice_list.sort()
    sorted_score_list = [avg_dict[idx] for idx in slice_list]
    moving_dict = {idx : _get_center_mean(sorted_score_list, idx)
                   for idx in range(len(sorted_score_list))}
    return moving_dict


def _get_moving_avg(avg_dict):
    slice_list = list(avg_dict.keys())
    slice_list.sort()
    sorted_score_list = [avg_dict[idx] for idx in slice_list]
    moving_sum = 0
    moving_max = 0
    for idx in range(len(sorted_score_list)):
        current_score = _get_center_mean(sorted_score_list, idx)
        moving_sum += current_score
        if moving_max <= current_score:
            moving_max = current_score
    moving_avg = moving_sum / len(sorted_score_list)
    return moving_avg, moving_max


def _get_center_mean(score_list, idx):
    score_sum = 0
    count = 0
    for i in range(3):
        if idx + (i - 1) not in range(len(score_list)):
            continue
        else:
            score_sum += score_list[idx + (i - 1)]
            count += 1
    avg = score_sum / (count * 1.0)
    return avg


def _check_merge_pairwise(b1, b2, label_combine_matrix, least_inter_ratio_matrix):
    x1, y1, w1, h1, slice1, l1, s1 = b1
    x2, y2, w2, h2, slice2, l2, s2 = b2
    label_combined = label_combine_matrix[l1][l2]
    least_inter_ratio = least_inter_ratio_matrix[l1][l2]
    if label_combined < 0:
        return False
    inter_rect = rect_intersect((x1, y1, w1, h1), (x2, y2, w2, h2))
    if not check_rect(inter_rect):
        return False
    min_area = min(w1 * h1, w2 * h2)
    if (inter_rect[2] * inter_rect[3] * 1.0 / min_area) < least_inter_ratio:
        return False

    if l1 == 11:
        c1 = [x1 + w1/2, y1 + h1/2]
        c2 = [x2 + w2/2, y2 + h2/2]
        c_dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        if w1 * h1 > w2 * h2 and c_dist > max(w2/2, h2/2):
            return False
        if w1 * h1 < w2 * h2 and c_dist > max(w1/2, h1/2):
            return False
    return True


def node_min_radius_thresh(bound_group, spacing):

    radius_thresh_nodule = 10
    radius_thresh_mass = 30

    nodule_flag = False
    for roi in bound_group[1]:
        # spacing(z, y, x), size(width, height)
        center, size, angle = cv2.minAreaRect(roi['edge'])
        if min(size) * spacing[1] > radius_thresh_mass:
            return True, True
        if min(size) * spacing[1] > radius_thresh_nodule:
            nodule_flag = True
    return nodule_flag, False


# ---------------------------------------------------------------------------------------------
# utils for reading full info
# ---------------------------------------------------------------------------------------------
def load_string_list(file_path, is_utf8=False):
    """
    Load string list from mitok file
    """
    try:
        with open(file_path, encoding='latin-1') as f:
            if f is None:
                return None
            l = []
            for item in f:
                item = item.strip()
                if len(item) == 0:
                    continue
                l.append(item)
    except IOError:
        print('open error %s' % file_path)
        return None
    else:
        return l


def load_full_info_file(file_path, is_utf8=True, is_filter_error=True):
    full_info_str_list = load_string_list(file_path, is_utf8)
    full_info_dict_list = []
    for line in full_info_str_list:
        items = line.split('\t')
        if items[0] == '[Valid]' and len(items) < 20:
            print(len(items))
            print(items)
            print('%s may be the old version, exit.' % file_path)
            return None
        if is_filter_error and items[0] == '[Valid]':
            is_valid, sub_dir, slice_thickness, slice_number, pixel_spacing, kvp, \
            current, kernel, manufacturer, series_date, power, \
            window_valus, img_size, ctdivol, body_part, series_desc, \
            img_pos, pat_pos, instance_range, slice_interval = items[:20]
            # img_pos, pat_pos, instance_range, slice_interval, spacing_between_slices = items[:21]
            img_pos_x, img_pos_y, img_pos_z = [float(x) for x in img_pos.split('\\')]
            slice_spacing = float(slice_interval)
            slice_thickness = float(slice_thickness)
            pixel_spacings = pixel_spacing.split(', ')
            pixel_spacings[0] = pixel_spacings[0][1:]
            pixel_spacings[-1] = pixel_spacings[-1][:-1]
            pixel_spacings = [float(x) for x in pixel_spacings]
            instance_min, instance_max = instance_range.split(', ')
            instance_min, instance_max = int(instance_min[1:]), int(instance_max[:-1])
            img_h, img_w = img_size.split(', ')
            img_h, img_w = int(img_h[1:]), int(img_w[:-1])
            d = {
                'sub_dir': sub_dir,
                'slice_spacing': slice_spacing,
                'slice_interval': slice_spacing,
                'kernel': kernel,
                'pixel_spacing': pixel_spacings,
                'slice_thickness': slice_thickness,
                'manufacturer': manufacturer,
                'patient_position': pat_pos,
                'ct_divol': None if ctdivol == '' else float(ctdivol),
                'kvp': None if kvp == '' else int(float(kvp)),
                'current': None if current == '' else int(float(current)),
                'instance_range': (instance_min, instance_max),
                'number_slices': int(slice_number),
                'image_size': (img_h, img_w),
                'image_position': (img_pos_x, img_pos_y, img_pos_z),
                'body_part': body_part
                # 'spacing_between_slices': spacing_between_slices
            }
            full_info_dict_list.append(d)
    return full_info_dict_list


def load_info_dict(info_file):
    info_list = load_full_info_file(info_file, is_utf8=False)

    info_dict = {}
    for info in info_list:
        info['patientID'], info['studyUID'], info['seriesUID'] = info['sub_dir'].split('/')[:3]
        info_dict[info['sub_dir']] = info
    info_dict = info_dict
    return info_dict


def coord2json(nodes):
    new_nodes = []
    for bounds_score in nodes:
        bounds, score = bounds_score
        new_bounds = []
        for bound in bounds:
            x0, y0, x1, y1 = bound['edge']
            edge = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            new_bounds.append({'edge': edge, 'slice_index': bound['slice_index']})
        new_nodes.append([new_bounds, score])
    return new_nodes


def write_results(sv_dir, merged_bounds, info):
    json_class = StandardJSON()
    json_data = json_class.make_json_result_from_detect_not_norm(info,
                                                                 coord2json(merged_bounds),
                                                                 json_format_version='2.0.0.180430',
                                                                 taskName='LungNoduleDetect', cut_float=2,
                                                                 norm_spacing=(0.6, 0.6, 0.6),
                                                                 display=False)
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    json_file = os.path.join(sv_dir, 'ann.json')
    json_class.write_json(json_file, json_data)


# ---------------------------------------------------------------------------------------------
# result filter
# ---------------------------------------------------------------------------------------------
def filter_by_lung_mask(boxes_array, rois_array, mask_dir, sub_dir):
    try:
        mask_npz = np.load(osp.join(mask_dir, sub_dir) + '/mask.npz')
    except:
        print(sub_dir, ' with no mask')
        return boxes_array, rois_array
    mask_array = mask_npz['data']
    # layer_num = mask_array.shape[0]

    z_tensor = mask_array.sum(-1).sum(-1)
    z_index = np.nonzero(z_tensor)
    z_min, z_max = z_index[0][0], z_index[0][-1]

    # x_top, y_top, z_top, w, h, d, 1, label, score
    # filter upper bound
    filter_flag = (boxes_array[:, 7] != 5) + (boxes_array[:, 7] == 5) * \
                  (boxes_array[:, 2] < z_min + (z_max - z_min)/2)
    boxes_f0 = boxes_array[filter_flag]
    rois_f0 = rois_array[filter_flag]

    # filter lower bound
    z_13 = z_min + (z_max - z_min)/2
    filter_flag = (boxes_f0[:, 7] != 13) * (boxes_f0[:, 7] != 12) + (boxes_f0[:, 7] == 13) * (boxes_f0[:, 2] < z_13) \
                  + (boxes_f0[:, 7] == 12) * (boxes_f0[:, 2] < z_max)
    boxes_f1 = boxes_f0[filter_flag]
    rois_f1 = rois_f0[filter_flag]

    return boxes_f1, rois_f1


def filter_by_image_feature(prediction, masks, image_path, slice_id):

    filter_rois = []
    keep = np.ones(len(masks))
    image = cv2.imread(image_path)
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        import pdb; pdb.set_trace()
    labels = prediction.get_field("labels")

    for idx, mask in enumerate(masks):

        # remove bbox wo contour, will cause lymph node FP raise
        if np.sum(mask.numpy()) == 0:
            keep[idx] = 0
            continue

        # filter thyroid nodule by image feature
        if labels[idx] == 5:
            roi_array = image * mask.numpy()[0]
            calc_ratio = np.sum(roi_array > 200)/np.sum(mask.numpy())
            if calc_ratio >= 0.15:
                keep[idx] = 0
                continue

        thresh = mask[0, :, :, None].type(torch.uint8)
        thresh = np.array(thresh * 255, dtype=np.uint8)
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # one box one roi
        # try! drop the box if no contour
        if not contours:
            tmp_cnt = [[0, 0]]
        else:
            tmp_cnt = max(contours, key=lambda coll: len(coll))
        filter_rois.append(tmp_cnt)

    keep = keep.nonzero()[0]
    return filter_rois, prediction[keep]

"""
def filter_by_organ_mask_ori(bound_groups, sub_dir, flip_flag):

    filter_bound_groups = []
    other_bound_groups = []
    mask_root = '/data/ms_data/segment/test/organ_mask'
    mask_path = osp.join(mask_root, sub_dir) + '/pred.nrrd'
    try:
        data, options = nrrd.read(mask_path)
    except:
        print(sub_dir.split('/')[0], ' has no organ mask')
        return bound_groups, other_bound_groups
    organ_mask = np.transpose(data, (2, 0, 1))
    organ_mask = np.flip(organ_mask, 0)

    inds_heart = np.where(organ_mask == 1)
    bottom = inds_heart[0].max()

    inds_aorta = np.uint8(organ_mask == 2)
    inds_pa = np.uint8(organ_mask == 5)

    for bound_group in bound_groups:
        valid_flag = True
        label = bound_group[0][0][5]

        # filter by heart
        label_list_heart = [1, 2, 3]
        if label in label_list_heart:
            center_idx = int(len(bound_group)/2)
            slice_idx = bound_group[center_idx][0][4]
            if slice_idx >= bottom + 3:
                valid_flag = False

        # filter by aorta or pa, without 15 in fact
        label_list_artery = [11, 15]
        if label in label_list_artery:
            center_idx = int(len(bound_group)/2)
            slice_idx = bound_group[center_idx][0][4]
            ratio_aorta = inter_ratio_calc(inds_aorta[slice_idx], bound_group[center_idx][1])

            ratio_pa = inter_ratio_calc(inds_pa[slice_idx], bound_group[center_idx][1])
            if ratio_aorta >= 0.8 or ratio_pa >= 0.7:
                valid_flag = False

        if valid_flag:
            filter_bound_groups.append(bound_group)
        if not valid_flag and label in label_list_heart:
            other_bound_groups.append(bound_group)

    return filter_bound_groups, other_bound_groups
"""


def filter_by_organ_mask(bound_groups, mask_array, sub_dir, flip_flag):
    # apply organ mask, heart 1, aorta 2, trachea 3, pa 4

    filter_bound_groups = []
    exclude_bound_groups = []

    mask_organs = np.uint8((mask_array >= 1) & (mask_array <= 3))
    mask_organs = torch.tensor(mask_organs)

    mask_pa = np.uint8(mask_array == 4)
    mask_pa = torch.tensor(mask_pa)

    # retrieve mask from mask array
    mask_heart = np.uint8(mask_array == 1)
    mask_heart = torch.tensor(mask_heart)

    inds_heart = np.where(mask_array == 1)
    gap = (inds_heart[0].max() - inds_heart[0].min()) / 3
    up_line = inds_heart[0].min() + gap
    mid_line = inds_heart[0].min() + 2 * gap
    btm_line = inds_heart[0].max()

    for bound_group in bound_groups:
        valid_flag = True
        label = bound_group[3]

        # filter by heart
        label_list_heart = [1, 2, 3]
        if label in label_list_heart:
            center_idx = int(len(bound_group[0])/2)
            slice_idx = bound_group[0][center_idx]["slice_index"]

            cardiac_pos = 'btm'
            if slice_idx <= up_line:
                cardiac_pos = 'up'
            elif slice_idx <= mid_line:
                cardiac_pos = 'mid'
            elif slice_idx <= btm_line:
                cardiac_pos = 'btm'
            else:
                valid_flag = False

            # affect 2 strongly, suggested by xiehe
            if label == 2 and valid_flag and cardiac_pos == 'btm':
                ratio_organ = iolesion_calc(mask_heart[slice_idx], bound_group[1][center_idx]['edge'], '2o1')
                if ratio_organ < torch.tensor(0.15):
                    valid_flag = False

        # filter by aorta or pa, without 15 in fact
        label_list_artery = [11, 15]
        if label in label_list_artery:
            center_idx = int(len(bound_group[0])/2)
            slice_idx = bound_group[0][center_idx]["slice_index"]
            ratio_organ = iolesion_calc(mask_organs[slice_idx], bound_group[1][center_idx]['edge'], 'iol')
            if ratio_organ >= torch.tensor(0.7):
                valid_flag = False

            ratio_all = []
            for b_th in range(len(bound_group[0])):
                slice_idx = bound_group[0][b_th]["slice_index"]
                ratio_all.append(iolesion_calc(mask_pa[slice_idx], bound_group[1][b_th]['edge'], 'iol'))
            ratio_sum_pa = torch.sum(torch.tensor(ratio_all) > torch.tensor(0.7))
            if 2 <= ratio_sum_pa <= len(bound_group[0]) / 3:
                valid_flag = False
                # print(label, "filtered by slice inter ratio with pa > 0.7")

        #  filter under heart, good for 12, affect 2
        if valid_flag and label != 4:
            center_idx = int(len(bound_group[0]) / 2)
            slice_idx = bound_group[0][center_idx]["slice_index"]
            if slice_idx > btm_line:
                valid_flag = False
                # print(label, "filtered by heart bottom line")

        if valid_flag:
            filter_bound_groups.append(bound_group)
        if not valid_flag and label in label_list_artery:
            exclude_bound_groups.append(bound_group)
    return filter_bound_groups, exclude_bound_groups


def iolesion_calc(img1, img2_contour, type):

    blank = np.zeros(img1.shape, np.uint8)
    img2 = cv2.drawContours(blank.copy(), [img2_contour], 0, 1, -1)
    img2 = torch.tensor(img2)
    # cv2.imwrite('drawing1.png', img1 * 255)
    # cv2.imwrite('drawing2.png', img2 * 255)

    if type == '2o1':
        area_1 = img1.sum().to(torch.float)
        area_2 = img2.sum().to(torch.float)
        ratio = area_2 / area_1
    elif type == 'iol':
        inter = img1 & img2
        area_i = inter.sum().to(torch.float)
        area_2 = img2.sum().to(torch.float)
        ratio = area_i / area_2
    else:
        print('wrong ratio')
        ratio = -1
    return ratio


