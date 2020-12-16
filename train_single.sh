CUDA_VISIBLE_DEVICES=4 python ./tools/train_net.py --config-file ms_mask_rcnn_R_50_FPN_3dce_mod.yaml SOLVER.IMS_PER_BATCH 2 DATALOADER.NUM_WORKERS 1
