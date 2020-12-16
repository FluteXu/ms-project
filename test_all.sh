path="/home/wangcheng/maskrcnn-benchmark/ckpt/test_all/"
log_dir="/home/wangcheng/maskrcnn-benchmark/log_results/"
gpu_id="0,1,2,3,4,5,6,7"
RANDOM=$$

log_name=.log
train_log=train.log
if [ ! -d $log_dir ];then
mkdir $log_dir
fi
files=$(ls $path)
for filename in $files
do
  echo $filename
  CUDA_VISIBLE_DEVICES=$gpu_id python -m torch.distributed.launch --nproc_per_node=8 --master_port=$RANDOM ./tools/test_net.py \
  --config-file ms_mask_rcnn_R_50_FPN_1x.yaml  \
  --ckpt $path$filename > $log_dir$filename$log_name 2>&1 \

done
