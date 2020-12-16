python ./pred_to_eval/pred_to_eval_format.py --subset coco_test_all --thresh-2d 0.05 \
--config-file ms_mask_rcnn_R_50_FPN_3dce_mod.yaml \
--image-root /data/ms_data/npz/test \
--png-root /data/ms_data/3d_slice_origin/test \
--lung-mask-dir /data/ms_data/segment/test/lung_mask \
--organ-mask-dir /data/ms_data/segment/test/organ_mask
