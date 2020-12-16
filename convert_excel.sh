python ./pred_to_eval/pred_to_excel_format.py --subset coco_test_qilu --thresh-2d 0.05 \
--config-file ms_mask_rcnn_R_50_FPN_3dce_mod.yaml \
--image-root /data/ms_data/npz/qilu \
--png-root /data/ms_data/3d_slice_origin/qilu \
--lung-mask-dir /data/ms_data/segment/qilu/lung_mask \
--organ-mask-dir /data/ms_data/segment/qilu/organ_mask
