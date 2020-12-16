import mitok.utils.rect
import mitok.utils.cube
import mitok.utils.union_find
import mitok.utils.mio
import numpy as np
from mitok.utils.merror import MError


class Patch3D(object):
    """
    Patch3D is defined as
    (x, y, z, w, h, d, t(direction), label, score, score1, score2, score3)
    """
    def __init__(self, cube, direct, label, score):
        self.err_str = ''
        self.is_valid = True
        self.cube = cube
        self.direct = direct
        self.label = label
        self.score = score
        if type(direct) != int or direct != 0:
            self.err_str = '[ERROR] direct != 0'
            self.is_valid = False
            return
        if not mitok.utils.cube.check_cube(cube):
            self.err_str = '[ERROR] wrong cube'
            self.is_valid = False
            return
        if label < 0 or score < 0:
            self.err_str = '[ERROR] label < 0 or score < 0'
            self.is_valid = False
            return

    def rescale(self, scales):
        scale_x, scale_y, scale_z = scales
        x, y, z, w, h, d = self.cube
        self.cube = [int(x * scale_x), int(y * scale_y), int(z * scale_z),
                     int(w * scale_x), int(h * scale_y), int(d * scale_z)]


class Bound3D(object):
    """
    Bound3D is used for nodule detection bound.
    ##@member cube: (x, y, z, w, h, d)
    ##@member direct: 0
    ##@member label: label of the detection
    ##@member score: score of the detection
    ##@member is_init: is empty
    ##@member is_patch: is an unit patch. if this is an unit patch, then src_patches=[] and get_patch_num()=1
    ##@member src_patches: the bound3d is combined from a list of bound3d_s,
                            each of which has only 1 unit size in self.direct
    """
    def __init__(self, cube=(-1, -1, -1, -1, -1, -1), direct=0, label=-1, score=-1):
        self.err_str = ''
        self.is_valid = True
        if type(cube) not in [tuple, list] or len(cube) != 6:
            self.err_str = '[ERROR] cube is wrong'
            self.is_valid = False
            return
        for c in cube:
            if type(c) != int and type(c) != float and type(c) != np.float32:
                self.err_str = '[ERROR] cube is wrong'
                self.is_valid = False
                return
        self.cube = cube
        direct = 0
        self.direct = direct
        if type(label) != int:
            self.err_str = '[ERROR] label is not int'
            self.is_valid = False
            return
        self.label = label
        if type(score) != int and type(score) != float:
            self.err_str = '[ERROR] score is not int or float'
            self.is_valid = False
            return
        self.score = score
        self.is_init = (mitok.utils.cube.check_cube(cube) and label >= 0 and score >= 0)
        self.src_patches = []
        if self.is_init:
            self.src_patches.append(Patch3D(cube, direct, label, score))

    def __str__(self):
        s = 'cube: %s  ' % str(self.cube)
        s += 'patch_number=%d, score=%.4f, label=%d' % (self.get_patch_num(), self.score, self.label)
        return s

    def __repr__(self):
        s = 'cube: %s  ' % str(self.cube)
        s += 'patch_number=%d, score=%.4f, label=%d' % (self.get_patch_num(), self.score, self.label)
        return s

    @staticmethod
    def __check_valid(cube, direct, label, score):
        is_direct_valid = (direct == 0)
        return mitok.utils.cube.check_cube(cube) and label >= 0 and score >= 0 and is_direct_valid

    def __combine_score(self, score):
        self.score += score

    def append(self, cube, direct, label, score, label_combine_matrix):
        if self.is_init:
            label_combined = label_combine_matrix[self.label][label]
            if label_combined < 0:
                return
            self.cube = mitok.utils.cube.cube_union(self.cube, cube)
            self.__combine_score(score)
            if label_combined < 0 or label_combined >= label_combine_matrix.shape[0]:
                self.is_valid = False
                return
                # return MError(MError.E_FIELD_BOUND, 2, '[ERROR] wrong label_combined'), None
            self.label = label_combined
        else:
            self.cube = cube
            self.direct = direct
            self.label = label
            self.score = score
            self.is_init = True
        self.src_patches.append(Patch3D(cube, direct, label, score))
        return
        # return MError(MError.E_FIELD_BOUND, 0, ''), True

    def get_score(self):
        return self.score

    def get_patch_num(self):
        return len(self.src_patches)

    def __dcm_to_physic_axis(self, point, origin_point, physic_direct, pixel_spacing_3d):
        origin_point = np.array(origin_point)
        # print(point, origin_point, self.pixel_spacing_3d)
        return origin_point + physic_direct * np.array(point) * pixel_spacing_3d

    def __trans_xyz_to_dcm(self, point, pixel_spacing_3d, norm_rate=0.6):
        return np.array(point) * norm_rate / pixel_spacing_3d

    def __convert_to_physics_axis(self, point, pixel_spacing_3d, origin_point,
                               physic_direct, norm_rate=0.6):
        dcm_axis = self.__trans_xyz_to_dcm(point, pixel_spacing_3d, norm_rate)
        # print('dcm axis', dcm_axis)
        physic_axis = self.__dcm_to_physic_axis(dcm_axis, origin_point,
                                                physic_direct, pixel_spacing_3d)
        return physic_axis

    def convert_to_physics_axis(self, point, pixel_spacing_3d, origin_point,
                               physic_direct, norm_rate=0.6):
        dcm_axis = self.__trans_xyz_to_dcm(point, pixel_spacing_3d, norm_rate)
        # print('dcm axis', dcm_axis)
        physic_axis = self.__dcm_to_physic_axis(dcm_axis, origin_point,
                                                physic_direct, pixel_spacing_3d)
        return MError(MError.E_FIELD_BOUND, 0, ''), physic_axis

    def convert_to_xyz_axis(self, point, pixel_spacing_3d, origin_point,
                               physic_direct, norm_rate=0.6):
        if len(point) != 3 or not self.__is_number(point[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'point error'), None
        if type(pixel_spacing_3d) != np.ndarray or len(pixel_spacing_3d) != 3 \
                or not self.__is_number(pixel_spacing_3d[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'pixel_spacing_3d error'), None
        if type(origin_point) != np.ndarray or len(origin_point) != 3 \
                or not self.__is_number(origin_point[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'origin_point error'), None
        if type(physic_direct) != np.ndarray or len(physic_direct) != 3 \
                or not self.__is_number(physic_direct[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'physic_direct error'), None

        dcm_axis = self.__trans_xyz_to_dcm(point, pixel_spacing_3d, norm_rate)
        # print('dcm axis', dcm_axis)
        return MError(MError.E_FIELD_BOUND, 0, ''), dcm_axis

    def get_center_physic_axis(self, pixel_spacing_3d, origin_point,
                               physic_direct, norm_rate=0.6):
        if type(pixel_spacing_3d) != np.ndarray or len(pixel_spacing_3d) != 3 \
                or not self.__is_number(pixel_spacing_3d[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'pixel_spacing_3d error'), None
        if type(origin_point) != np.ndarray or len(origin_point) != 3 \
                or not self.__is_number(origin_point[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'origin_point error'), None
        if type(physic_direct) != np.ndarray or len(physic_direct) != 3 \
                or not self.__is_number(physic_direct[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'physic_direct error'), None
        x, y, z, w, h, d = self.cube
        center_ori = [x + w / 2.0, y + h / 2.0, z + d / 2.0]
        dcm_axis = self.__trans_xyz_to_dcm(center_ori, pixel_spacing_3d, norm_rate)
        physic_axis = self.__dcm_to_physic_axis(dcm_axis, origin_point,
                                                physic_direct, pixel_spacing_3d)
        return MError(MError.E_FIELD_BOUND, 0, ''), physic_axis

    def __is_number(self, a):
        try:
            x = float(a)
            return True
        except:
            return False

    def output(self, pixel_spacing_3d, origin_point,
                physic_direct, norm_rate=0.6):
        if type(pixel_spacing_3d) != np.ndarray or len(pixel_spacing_3d) != 3 \
                or not self.__is_number(pixel_spacing_3d[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'pixel_spacing_3d error'), None
        if type(origin_point) != np.ndarray or len(origin_point) != 3 \
                or not self.__is_number(origin_point[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'origin_point error'), None
        if type(physic_direct) != np.ndarray or len(physic_direct) != 3 \
                or not self.__is_number(physic_direct[0]):
            return MError(MError.E_FIELD_BOUND, 1, 'physic_direct error'), None
        x, y, z, w, h, d = self.cube
        physic_axis1 = self.__convert_to_physics_axis((x, y, z),
                            pixel_spacing_3d, origin_point,
                            physic_direct, norm_rate)
        x2 = x + w
        y2 = y + h
        z2 = z + d
        physic_axis2 = self.__convert_to_physics_axis((x2, y2, z2),
                                                      pixel_spacing_3d, origin_point,
                                                      physic_direct, norm_rate)
        whd = physic_axis2 - physic_axis1
        cube_new = [physic_axis1[0], physic_axis1[1], physic_axis1[2],
                    whd[0], whd[1], whd[2]]
        data = dict()
        data['cube'] = cube_new
        data['label'] = self.label
        data['score'] = self.score
        return MError(MError.E_FIELD_BOUND, 0, ''), data


def __can_two_combined(b1, b2, label_combine_matrix, least_inter_ratio_matrix):
    """
    :param b1: bound1
    :param b2: bound2
    :param label_combine_matrix: k*k numpy array. If b1/b2 has label l1/l2,
        m[l1][l2] is the label for combined bound.  m[l1, l2]=-1 if cannot be combined.
    :param least_inter_ratio_matrix: we combine 2 bounds only when the
        intersect_area/min_bound_area >= least_inter_ratio_matrix[l1][l2]
    :return: whether two bounds can be combined.
    """
    x1, y1, w1, h1, slice1, l1, s1 = b1
    x2, y2, w2, h2, slice2, l2, s2 = b2
    label_combined = label_combine_matrix[l1][l2]
    least_inter_ratio = least_inter_ratio_matrix[l1][l2]
    if label_combined < 0:
        return False
    inter_rect = mitok.utils.rect.rect_intersect((x1, y1, w1, h1), (x2, y2, w2, h2))
    if not mitok.utils.rect.check_rect(inter_rect):
        return False
    min_area = min(w1 * h1, w2 * h2)
    if (inter_rect[2] * inter_rect[3] * 1.0 / min_area) < least_inter_ratio:
        return False
    return True


def check_opt(opt):
    keys = ['label_combine_matrix',
            'least_inter_ratio_matrix',
            'max_slices_stride']
    if type(opt) != dict:
        return False
    for k in keys:
        if k not in opt:
            return False
    val = opt['label_combine_matrix']
    if type(val) != np.ndarray or val.shape != (3,3):
        return False
    val = opt['least_inter_ratio_matrix']
    if type(val) != np.ndarray or val.shape != (3, 3):
        return False
    val = opt['max_slices_stride']
    if type(val) != int:
        return False
    return True

def cluster_bounds_within_each_slice(bounds_raw, direct, opt):
    """
    add by zzhou
    reduce the number of patches within slices, especially useful for multi-nodule
    """
    from sklearn.cluster import MeanShift
    new_bounds_raw = []
    
    # prepare data
    bound2ds_dict = {}
    patch_dict = {}
    for i in range(bounds_raw.shape[0]):
        score = bounds_raw[i, 8]
        x, y, z, w, h, d, direct_, label = list(map(int, bounds_raw[i, :8]))
        z += d // 2
        bound2ds_dict.setdefault(z, [])
        bound2ds_dict[z].append(bounds_raw[i])
        patch_dict.setdefault(z, [])
        patch_dict[z].append([x + w/2, y + h/2])
    
    # cluster using mean shift 
    for slice_id in patch_dict:
        clf = MeanShift(bandwidth=opt['bandwidth'], cluster_all=True).fit(patch_dict[slice_id])
        labels = clf.labels_
        centers = clf.cluster_centers_
        # use an average patch instead multiple patches within each cluster
        for idx in xrange(centers.shape[0]): 
            center = centers[idx]
            label_list = []
            #w, h, d, score = 0, 0, 0, 0
            score = 0
            count = 0
            num_small, num_big = 0, 0
            size_small, size_big = int(min(opt['patch_size'])), int(max(opt['patch_size']))
            for l, label in enumerate(labels):
                if label == idx:
                    score += bound2ds_dict[slice_id][l][8] 
                    _x, _y, _z, _w, _h, _d, _direct, _label = list(map(int, bound2ds_dict[slice_id][l][:8]))
                    if _w == size_small:
                        num_small += 1
                    elif _w == size_big:
                        num_big += 1
                    #w += _w
                    #h += _h
                    #d += _d
                    #count += 1
                    label_list.append(_label)
            label = min(label_list)
            if num_big > num_small:
                w = h = d = size_big 
            else:
                w = h = d = size_small
            #if count > 0:
            #    w =  int(w / count)
            #    h = int(h / count)
            #    d = int(d / count)
            x, y = max(0, int(round(center[0])) - w/2), max(0, int(round(center[1])) - h/2)
            new_bounds_raw.append([x, y, slice_id - d/2, w, h, d, direct, label, score])
            #new_bounds_raw.append([int(round(center[0])), int(round(center[1])), slice_id - d/2, w, h, d, direct, label, score])
    return np.array(new_bounds_raw)

def combine_bounds_3d_depend_on_numbers(bounds_raw, mask_tensor, direct, opt):

    mask_xy = np.nonzero(mask_tensor)
    z_min = mask_xy[0].min()
    z_max = mask_xy[0].max()
    y_min = mask_xy[1].min()
    y_max = mask_xy[1].max()
    x_min = mask_xy[2].min()
    x_max = mask_xy[2].max()
    mask_w, mask_h, mask_d = x_max - x_min + 1.0, y_max - y_min + 1.0, z_max - z_min + 1.0
    
    multi_nodule = False
    err, bound3ds = combine_bounds_3d(bounds_raw, direct, opt)
    for bound3d in bound3ds:
        _, _, _, w, h, d = bound3d.cube
        #if w / mask_w > opt['ratio'][0] or h / mask_h > opt['ratio'][1] or d / mask_d > opt['ratio'][2]:
        if w / mask_w >= opt['ratio'] or h / mask_h >= opt['ratio']:
            multi_nodule = True
            break
    if multi_nodule:
        print( 'Detect multiple nodules, processing...')
        print( 'before cluster, #patches =', len(bounds_raw))
        bounds_raw = cluster_bounds_within_each_slice(bounds_raw, direct, opt)
        print( 'after cluster, #patches =', len(bounds_raw))
        err, bound3ds = combine_bounds_3d(bounds_raw, direct, opt)
    
    return err, bound3ds, multi_nodule

def combine_bounds_3d(bounds_raw, direct, opt):
    """
    :param bounds_raw:
    :param direct:
    :param opt: see more in combine_bounds_3d_direct
    :return:
    """
    if type(bounds_raw) != np.ndarray:
        return MError(MError.E_FIELD_BOUND, 1, '[ERROR] bounds type invalid'), None
    shape = bounds_raw.shape
    if len(shape) != 2 or shape[1] < 9:
        return MError(MError.E_FIELD_BOUND, 1, '[ERROR] bounds shape invalid'), None
    if not check_opt(opt):
        return MError(MError.E_FIELD_BOUND, 1, '[ERROR] opt invalid'), None
    label_combine_matrix = opt['label_combine_matrix']
    if bounds_raw.shape[0] == 0:
        return MError(MError.E_FIELD_BOUND, 0, ''), []

    # 1) turn all bounds_raw into a list of bound2ds in given direction
    bound2ds_list = []
    for i in range(bounds_raw.shape[0]):
        score = bounds_raw[i, 8]
        x, y, z, w, h, d, direct_, label = list(map(int, bounds_raw[i, :8]))
        z += d // 2
        bound2ds_list.append([x, y, w, h, z, label, score])

    # 2)
    try:
        bound_groups = combine_bounds_3d_direct(bound2ds_list, opt)
    except:
        return MError(MError.E_FIELD_BOUND, 2, '[ERROR] combine_bounds_3d_direct'), None
    # 3) generate Bound3D using bound_groups and direct
    bound3ds = []
    try:
        for bound_group in bound_groups:
            bound3d = Bound3D()
            for bound2d in bound_group:
                x, y, w, h, slice_id, label, score = bound2d
                cube = [x, y, slice_id, w, h, 1]
                bound3d.append(cube, direct, label, score, label_combine_matrix)
            if not bound3d.is_valid:
                return MError(MError.E_FIELD_BOUND, 3, 'Bound3d invalid'), None
            bound3ds.append(bound3d)
    except Exception as ex:
        return MError(MError.E_FIELD_BOUND, 2, '[ERROR] generate Bound3D: %s' % ex), None

    return MError(MError.E_FIELD_BOUND, 0, ''), bound3ds


def combine_bounds_3d_direct(bound2ds_list, opt):
    """
    TODO: think of the score between the
    :param bound2ds_list: list of bounds of all slices. bounds[i] contains all bounds2d in slice_i,
        which is n*6 list array, each row defines as [x, y, w, h, slice_id, label, score]
    :param opt: options for combining
                label_combine_matrix: k*k numpy array. If b1/b2 has label l1/l2,
                        m[l1][l2] is the label for combined bound.  m[l1, l2]=-1 if cannot be combined.
                least_inter_ratio_matrix: we combine 2 bounds only when the
                        intersect_area/min_bound_area >= least_inter_ratio_matrix[l1][l2]
                max_slices_stride: when combined, we can at most skip max_slices_stride-1 slices
                        if the bounds are not continuous
    :return: bound2d
    """
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
        combine_pairs.append((i, ))
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
                    if __can_two_combined(b1, b2, label_combine_matrix, least_inter_ratio_matrix):
                        combine_pairs.append((id1, id2))

    # 3) union find
    uf = mitok.utils.union_find.UnionFind(combine_pairs)
    combined_bound_ids = uf.run()

    # 4) combine the bounds
    bound_groups = []
    for bound_ids in combined_bound_ids:
        bound_group = []
        for b_id in bound_ids:
            bound_group.append(bound2ds_list[b_id])
        bound_groups.append(bound_group)

    # May need to check all bounds again
    # TODO

    return bound_groups
