CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
./tools/test_net.py --config-file ms_mask_rcnn_R_50_FPN_3dce_mod.yaml \
--ckpt ./ms_mask_rcnn_R_50_FPN_3dce_mod/model_final.pth
