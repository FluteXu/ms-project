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


name_cls_map = {
                '心包增厚': 1,
                '心包积液': 2,
                '心包钙化': 3,
                '食管裂孔疝': 4,
                '甲状腺结节': 5,
                '冠状动脉钙化/支架影': 6,
                '主动脉壁间血肿': 7,
                '主动脉内膜钙化移位': 8,
                '二尖瓣钙化': 9,
                '主动脉瓣钙化': 10,
                '纵隔结节肿块': 11,
                '膈淋巴肿': 12,
                '腋窝淋巴肿': 13,
                '內乳淋巴肿': 14,
                }


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


def dump_predictions(results_det, results_roi, thresh_2d, mask_dir, save_root, image_root,
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

    global pos_count
    global neg_count

    # 14 classes
    thresh_3d = [0] + [0.05] * 14
    min_merged_boxes = [0, 3, 3, 3, 4, 3, 2, 2, 2, 2, 2, 4, 4, 4, 4]

    # [bound2d, roi2d, avg_score, label]
    filter_bound_groups = []
    for bound_group in merged_bound3d:
        thresh_flag = True

        # cls 0 is background
        index = bound_group[3]
        if len(bound_group[0]) < min_merged_boxes[index]:
            # print('in thresh_merged_bound3d: ', len(bound_group), min_merged_boxes[cls])
            continue

        if bound_group[2] > thresh_3d[index] and thresh_flag:
            filter_bound_groups.append(bound_group)
    return filter_bound_groups


"""
def thresh_merged_bound3d_ori(merged_bound3d, spacing, pid):

    thresh_3d = [0] + [0.05] * 14
    min_merged_boxes = [0, 3, 3, 3, 4, 3, 2, 2, 2, 2, 2, 4, 4, 4, 4]

    # [bound2d, roi2d, avg_score, label]
    filter_bound_groups = []
    for bound_group in merged_bound3d:
        thresh_flag = True

        # cls 0 is background
        cls = bound_group[0][0][5]
        if len(bound_group) < min_merged_boxes[cls]:
            continue

        if get_score_one(bound_group) > thresh_3d[cls] and thresh_flag:
            filter_bound_groups.append(bound_group)
    return filter_bound_groups
"""


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


def write_excel(df, out_dir, sheetname='Sheet1', write_index=False, file_name='CT_eval.xlsx'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, file_name)
    writer = pd.ExcelWriter(fname, engine='xlsxwriter')
    df.to_excel(writer, sheetname, index=write_index)
    writer.save()


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

    # model = args.model.lower()
    # result_file = osp.join(save_dir, 'predictions.pth')
    # results_det, results_roi = post_process_results(cfg, result_file, args.subset, args.png_root)
    # save_pkl(results_det, results_roi, save_dir)

    results_det, results_roi = load_pkl(save_dir)
    dump_predictions(results_det, results_roi, args.thresh_2d, args.lung_mask_dir, save_dir,
                     args.image_root, args.organ_mask_dir, vis_filter_flag)


if __name__ == '__main__':
    main()
