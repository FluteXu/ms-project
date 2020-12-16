import os
import pickle
import os.path as osp


def main():

    all_results_dict = {}
    for sub_task in combine_list:
        det_file = osp.join(ct_det_root, sub_task, 'CT_detections.pkl')

        with open(det_file, 'rb') as f:
            ct_det_dict = pickle.load(f)

            for k, v in ct_det_dict.items():
                if k in all_results_dict.keys():
                    print("ERROR: duplicated sub_dir: ", k)
                    # import pdb; pdb.set_trace()
                else:
                    all_results_dict[k] = v

    save_object(all_results_dict, save_path)


def save_object(obj, file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    ct_det_root = '/home/wangcheng/maskrcnn-benchmark/inference'
    save_path = osp.join(ct_det_root, 'CT_detections.pkl')
    combine_list = ['coco_test1', 'coco_test2']

    main()
