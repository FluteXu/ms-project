CUDA_VISIBLE_DEVICES=0 python ./tools/test_net.py \
--config-file ms_mask_rcnn_R_50_FPN_3dce_mod.yaml \
--ckpt ./ms_mask_rcnn_R_50_FPN_3dce_mod_resnet/model_final.pth \
TEST.IMS_PER_BATCH 2 DATALOADER.NUM_WORKERS 1
