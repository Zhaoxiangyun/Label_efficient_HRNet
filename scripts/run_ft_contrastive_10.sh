RND_PORT=`shuf -i 4000-7999 -n 1`
PY_CMD="python -m torch.distributed.launch --master_port $RND_PORT --nproc_per_node=1"

CUDA_VISIBLE_DEVICES=3 $PY_CMD tools/train.py --cfg experiments/cityscapes/ft_contrastive_10.yaml

