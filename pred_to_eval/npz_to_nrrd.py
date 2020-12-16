import os
import os.path as osp
import nrrd
import numpy as np
from glob import glob


def npz_to_nrrd(root_path, npz_name):

    sub_dirs = glob(root_path + '*/*/*/')
    count = 0
    for sub_dir in sub_dirs:
        if count > 6:
            continue
        data = np.load(osp.join(sub_dir, npz_name))

        if npz_name == 'image.npz':
            flip = bool(data['flip'])
        else:
            flip = False

        if flip:
            tmp = np.flip(data['data'], axis=0)
        else:
            tmp = data['data']
        tmp = np.transpose(tmp, (2, 1, 0))

        dest = osp.join(sub_dir, 'tmp.nrrd')
        nrrd.write(dest, tmp)

        print(count, sub_dir)
        count += 1


if __name__ == '__main__':
    npz_root = '/data2/ms_data/npz/test/'
    mask_root = '/data2/ms_data/segment/test/'

    # npz_to_nrrd(npz_root, 'image.npz')
    npz_to_nrrd(mask_root, 'mask.npz')
