PY_CMD="python -m torch.distributed.launch --nproc_per_node=2"
RND_PORT=`shuf -i 4000-7999 -n 1`

CUDA_VISIBLE_DEVICES=0,1 $PY_CMD --master_port $RND_PORT tools/train.py --cfg experiments/cityscapes/seg_hrnet_second_0_05.yaml

