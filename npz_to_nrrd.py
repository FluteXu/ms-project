import os
import os.path as osp
import numpy as np
import nrrd
import glob

"""

organ mask is flipped, npz image is zyx, mask is yxz
image npz zyx -> xyz
organ mask 1. flip  2. yxz -> xyz (trans102)

"""


if __name__ == '__main__':

    pid = '0818353'
    root_dir = '/data/ms_data/npz/test'
    pid_dir = osp.join(root_dir, pid)

    # retrieve the first ss
    ss_dir = glob.glob(pid_dir + '/*/*')[0]
    file_path = osp.join(ss_dir, 'image.npz')
    save_path = osp.join(ss_dir, 'image.nrrd')

    # for itk-snap [x, y, z]
    # head is slice 0, at the bottom
    img = np.load(file_path)  # [z, y, x]
    print('npz shape: ', img['data'].shape)
    print('flip flag in npz: ', img['flip'])

    # slice 0 is head
    # flip flag in npz is false
    img_data = np.transpose(img['data'], (2, 1, 0))
    nrrd.write(save_path, img_data)
