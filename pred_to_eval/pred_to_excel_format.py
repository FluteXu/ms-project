import nrrd
import shutil
import argparse
import pandas as pd
from time import time
from PIL import Image, ImageFont, ImageDraw
from six.moves import cPickle as pickle
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.build import build_dataset
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, boxlist_merge
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from pred_to_eval.util import *
pil_font = ImageFont.truetype("pred_to_eval/simhei.ttf", 16)


pred_class = {
    u'心包增厚': [u'心包增厚', u'心包增厚(软组织)'],
    u'心包积液': [u'心包积液'],
    u'心包钙化': [u'心包钙化'],
    u'食管裂孔疝': [u'食管裂孔疝'],
    u'甲状腺结节': [u'甲状腺结节'],
    u'冠状动脉钙化/支架影': [u'冠状动脉钙化', u'冠状动脉支架影', u'冠状动脉钙化/支架影'],
    u'主动脉壁间血肿': [u'主动脉壁间血肿'],
    u'主动脉内膜钙化移位': [u'主动脉内膜钙化移位'],
    u'二尖瓣钙化': [u'二尖瓣钙化'],
    u'主动脉瓣钙化': [u'主动脉瓣钙化'],
    u'纵隔结节肿块': [u'上纵隔淋巴肿/肿块', u'前纵隔淋巴肿/肿块', u'中后纵隔淋巴肿/肿块',
                u'肺门淋巴肿/肿块（不含肺动脉）'],
    u'膈淋巴肿': [u'隔淋巴肿（隔顶到隔底）', u'膈淋巴肿（膈顶到膈底）'],
    u'腋窝淋巴肿': [u'腋窝淋巴肿'],
    u'內乳淋巴肿': [u'內乳淋巴肿'],
}

res_class = {
    1: u'心包增厚',
    2: u'心包积液',
    3: u'心包钙化',
    4: u'食管裂孔疝',
    5: u'甲状腺结节',
    6: u'冠状动脉钙化/支架影',
    7: u'主动脉壁间血肿',
    8: u'主动脉内膜钙化移位',
    9: u'二尖瓣钙化',
    10: u'主动脉瓣钙化',
    11: u'淋巴结肿大',
    12: u'纵隔肿块'
}

class_num = 15  # with background
select_class = [1, 2, 11, 12]
thresh_3d = [0] + [0.05] * (class_num - 1)
min_merged_boxes = [0, 3, 3, 3, 4, 3, 2, 2, 2, 2, 2, 4, 4, 4, 4]


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Convert prediction to Dr.Wise format')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        '--subset', dest='subset', required=False, default='coco_lg3',
        help='dataset name')
    parser.add_argument(
        '--lung-mask-dir', dest='lung_mask_dir', required=False,
        default='',
        help='path to lung mask')
    parser.add_argument(
        '--organ-mask-dir', dest='organ_mask_dir', required=False,
        default='',
        help='path to lung mask')
    parser.add_argument(
        '--image-root', dest='image_root', required=False,
        default='')
    parser.add_argument(
        '--png-root', dest='png_root', required=False,
        default='')
    parser.add_argument(
        '--root-dir', dest='root_dir', required=False,
        default='')
    parser.add_argument(
        '--model', dest='model', required=False, default='maskrcnn')
    parser.add_argument(
        '--thresh-2d', dest='thresh_2d',
        default=0.001,
        help='threshold for stage 1 detectors',
        type=float)
    return parser.parse_args()


def post_process_results(cfg, result_file, dataset, png_root):
    """
    one CT one results_dict/results[sub_dir]
    one image one predication
    one prediction contains several boxes with different
    classes
    """
    t = time()
    print("Processing results...")
    mask_threshold = 0.5
    masker = Masker(threshold=mask_threshold, padding=1)
    predictions = torch.load(result_file)

    dataset = build_dataset(cfg, dataset_list=[dataset], transforms=None,
                            dataset_catalog=DatasetCatalog, is_train=False)[0]

    results_det = {}
    results_roi = {}
    for i in range(len(dataset)):

        print(i)
        # filter 1
        # if i > 200:
        #     sys.exit()

        img_info = dataset.get_img_info(i)
        img_path = img_info['file_name']

        # filter 2
        # sid = img_path.split('/')[-2]
        # if sid != '':
        #     continue

        slice_str = img_path.split('/')[-1]
        sub_dir = img_path.replace('/' + slice_str, '')
        slice_id = int(slice_str.split('.')[0])
        image_width = img_info["width"]
        image_height = img_info["height"]

        # every image has a prediction, a prediction contains multiple bbox
        prediction = predictions[i]
        if len(prediction) == 0:
            continue

        prediction = boxlist_nms(prediction, nms_thresh=0.1)
        prediction = boxlist_merge(prediction)

        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xyxy')

        if prediction.has_field("mask"):
            masks = prediction.get_field("mask")
            masks = masker([masks], [prediction])[0]

            # filter the prediction according to image feature, i.e. thyroid nodule
            # roi_i: the i-th image contours of ct
            img_path = osp.join(png_root, img_path)
            rois_i, prediction = filter_by_image_feature(prediction, masks, img_path, slice_id)
        else:
            print("Error: prediction do not have mask!")
            sys.exit()

        if sub_dir not in results_roi:
            results_roi[sub_dir] = []
        results_roi[sub_dir].append(rois_i)

        dets_i = np.concatenate([
            prediction.bbox.numpy(),
            prediction.get_field("labels").numpy()[:, np.newaxis],
            prediction.get_field("scores").numpy()[:, np.newaxis],
        ], axis=1)
        res_slice = np.insert(dets_i, 4, slice_id, axis=1)
        res_slice = res_slice
        if sub_dir not in results_det:
            results_det[sub_dir] = []
        results_det[sub_dir].append(res_slice)

        # one box one mask
        assert len(rois_i) == dets_i.shape[0]
    print("Finished, cost {:.2f} sec.".format(time() - t))
    return results_det, results_roi


def save_object(obj, file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def gen_3d_pred(results_det, results_roi, thresh_2d, mask_dir, save_root, image_root,
                organ_mask_root, vis_filter_flag):
    print("Writing results to Dr.Wise format...")

    # create visualization directory
    vis_save_dir = '/home/wangcheng/maskrcnn-benchmark/vis_filter_bounds'
    if os.path.exists(vis_save_dir):
        shutil.rmtree(vis_save_dir, True)

    t = time()
    all_results_dict = {}
    for i, sub_dir in enumerate(results_det):

        # if sub_dir.split('/')[0] != '0818353':
        #     continue

        print('writing {}/{}'.format(i, len(results_det)))
        # retrieve image info
        pid = sub_dir.split('/')[0]
        image_path = osp.join(image_root, sub_dir) + '/image.npz'
        image_info = np.load(image_path)

        # flip flag indicate head-foot order
        flip_flag = image_info['flip']
        if flip_flag:
            print(sub_dir, 'is in Reverse Order!')
            import pdb; pdb.set_trace()

        # generate N*12 array and filter 2d result
        dets_ct, rois_ct = boxlist_to_arr12(results_det[sub_dir], results_roi[sub_dir])
        assert len(dets_ct) == len(rois_ct)
        # print('z_max for ', sub_dir, ' is ', len(dets_ct))

        # apply lung mask, must sorted by slice id first
        s1 = time()
        dets_ct, rois_ct = filter_by_lung_mask(dets_ct, rois_ct, mask_dir, sub_dir)
        print("filter_by_lung_mask time:", time() - s1)

        # merge 2d box to 3d
        if len(dets_ct) > 0:
            # merged_bounds = merge_bounds_3d_ori(dets_ct, rois_ct, thresh_2d, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            #                                 max_slices_stride=1, iom_thresh=0.7)
            merged_bounds = merge_bounds_3d(dets_ct, rois_ct, thresh_2d, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
                                            max_slices_stride=1, iom_thresh=0.7)
        else:
            merged_bounds = []

        # 3d result filter
        # bound_groups = thresh_merged_bound3d_ori(merged_bounds, image_info['spacing'], pid)
        bound_groups = thresh_merged_bound3d(merged_bounds, image_info['spacing'], pid)

        # apply organ mask, heart 1, aorta 2, trachea 3, pa 4
        mask_path = osp.join(organ_mask_root, sub_dir) + '/pred.nrrd'
        if os.path.exists(mask_path):
            data, options = nrrd.read(mask_path)
            # matching process
            data = np.flip(data, 2)
            mask_array = np.transpose(data, (2, 0, 1))

            s2 = time()
            # bound_groups, others = filter_by_organ_mask_ori(bound_groups, sub_dir, flip_flag)
            bound_groups, exclude_groups = filter_by_organ_mask(bound_groups, mask_array, sub_dir, flip_flag)
            print("filter_by_organ_mask time:", time() - s2)

            # ====== visualize filtered bound_groups ======
            if exclude_groups != [] and vis_filter_flag:
                visualize_3d(sub_dir, exclude_groups, vis_save_dir, image_root, mask_array)

        all_results_dict[sub_dir] = bound_groups
    ct_det_file = os.path.join(save_root, 'CT_detections.pkl')
    save_object(all_results_dict, ct_det_file)

    print("Finished, cost {:.2f} sec.".format(time() - t))


def thresh_merged_bound3d(merged_bound3d, spacing, pid):

    # bound_group: [bound2d, roi2d, avg_score, label]
    filter_bound_groups = []
    for bound_group in merged_bound3d:
        cls = bound_group[3]
        if len(bound_group[0]) < min_merged_boxes[cls]:
            continue

        # threshold mass and nodule size
        if cls in [11, 12, 13]:
            nodule_flag, mass_flag = node_min_radius_thresh(bound_group, spacing)
            # change label to mass
            if mass_flag:
                bound_group[3] = 15
        else:
            nodule_flag = True

        if bound_group[3] > thresh_3d[cls] and nodule_flag:
            filter_bound_groups.append(bound_group)
    return filter_bound_groups


def visualize_3d(sub_dir, merged_bounds, vis_save_dir, image_root, mask_array):

    img_path = osp.join(image_root, sub_dir) + '/image.npz'
    image_array = np.load(open(img_path, 'rb'))['data']
    mask_depth = image_array.shape[0]

    # select visual mask
    mask_array = np.uint8(mask_array == 5)
    mask_slice_info = gen_mask_slice(mask_array)
    pred_slice_info = gen_pred_slice(merged_bounds, mask_depth)
    draw_on_img(img_path, image_array, mask_slice_info, pred_slice_info, vis_save_dir)


def gen_pred_slice(merged_bounds, z_max):
    slice_info = [[] for _ in range(z_max)]
    group_id = 0
    for bound_group in merged_bounds:
        score = bound_group[2]
        label = bound_group[3]
        for bound in bound_group[1]:
            slice_idx = bound['slice_index']
            slice_info[slice_idx].append([bound['edge'], [group_id, label, score]])
        group_id += 1
    return slice_info


def gen_mask_slice(mask_array):
    slice_info = []
    for i in range(len(mask_array)):
        contours, hierarchy = cv2.findContours(mask_array[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        slice_info.append(contours)
    return slice_info


def draw_on_img(img_path, image_array, mask_slice_info, pred_slice_info, out_dir):

    def compute_colors_for_labels():

        label_len = len(name_cls_map)
        labels = np.array([i+1 for i in range(label_len)])
        palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = labels[:, None] * palette
        colors = (colors % 255).astype("uint8")

        return colors.tolist()

    def overlay_mask(image, rois):

        for roi in rois:
            try:
                contour = roi[0]
            except:
                import pdb; pdb.set_trace()
            color = color_array[roi[1][1]-1]
            image = cv2.drawContours(image, [contour], -1, color, 1)

        return image

    def add_pred_text(image, rois):

        text = ''
        for roi in rois:
            text += str(roi[1][0])+str(list(name_cls_map.keys())[roi[1][1]-1])+' '
        pil_im = Image.fromarray(np.uint8(image))
        draw = ImageDraw.Draw(pil_im)
        draw.text((5, 40), text, (255, 255, 255), font=pil_font)
        image = np.array(pil_im)
        return image

    pid, stid, ssid, img_name = img_path.split(os.sep)[-4:]
    save_dir = os.path.join(out_dir, pid, stid, ssid)
    color_array = compute_colors_for_labels()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for index in range(image_array.shape[0]):

        pred_bounds = pred_slice_info[index]
        mask_bounds = mask_slice_info[index]
        image = image_array[index]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if pred_bounds:
            image = overlay_mask(image.copy(), pred_bounds)
            if mask_bounds != []:
                for mask_bound in mask_bounds:
                    image = cv2.drawContours(image, mask_bound, -1, (0, 255, 0), 5)
            image = add_pred_text(image, pred_bounds)

            cv2.putText(image, 'pred', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            save_path = os.path.join(save_dir, str(index)+'.png')
            cv2.imwrite(save_path, image)


def gen_result_list(bound_groups):
    res_array = np.zeros(len(res_class))
    for bound_group in bound_groups:
        class_idx = bound_group[3]

        # label mapping
        if class_idx in [11, 12, 13]:
            class_idx = 11
        elif class_idx == 14:
            continue
        elif class_idx == 15:
            class_idx = 12

        if res_array[class_idx-1] == 0:
            res_array[class_idx-1] = 1

    return res_array.tolist()


def save_to_excel(results_dict, save_root):
    cols = ['patient_id']
    tmp = [*res_class.values()]
    cols.extend([tmp[i-1] for i in select_class])
    print(cols)

    excel = []
    for key, value in results_dict.items():
        current = []
        pid = key.split('/')[0]
        current.append(pid)
        class_ind = gen_result_list(value)
        class_ind = [class_ind[i-1] for i in select_class]
        current.extend(class_ind)
        excel.append(current)

    out_df = pd.DataFrame(excel, columns=cols)
    write_excel(out_df, save_root, file_name='qilu_model_result.xlsx')


def save_pkl(results_det, results_roi, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path1 = osp.join(save_path, 'CT_inter_det.pkl')
    save_object(results_det, save_path1)

    save_path2 = osp.join(save_path, 'CT_inter_roi.pkl')
    save_object(results_roi, save_path2)


def load_pkl(load_path):
    load_path1 = osp.join(load_path, 'CT_inter_det.pkl')
    with open(load_path1, 'rb') as f:
        results_det = pickle.load(f)

    load_path2 = osp.join(load_path, 'CT_inter_roi.pkl')
    with open(load_path2, 'rb') as f:
        results_roi = pickle.load(f)
    return results_det, results_roi


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    vis_filter_flag = False
    save_dir = osp.join('.', cfg.OUTPUT_DIR, 'inference', args.subset)

    result_file = osp.join(save_dir, 'predictions.pth')
    results_det, results_roi = post_process_results(cfg, result_file, args.subset, args.png_root)
    save_pkl(results_det, results_roi, save_dir)

    results_det, results_roi = load_pkl(save_dir)
    gen_3d_pred(results_det, results_roi, args.thresh_2d, args.lung_mask_dir, save_dir,
                     args.image_root, args.organ_mask_dir, vis_filter_flag)
    
    results_dict = pickle.load(open(osp.join(save_dir, 'CT_detections.pkl'), 'rb'))
    save_to_excel(results_dict, save_dir)


if __name__ == '__main__':
    main()
