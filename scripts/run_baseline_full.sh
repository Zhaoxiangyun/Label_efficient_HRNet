PY_CMD="python -m torch.distributed.launch --nproc_per_node=1"

CUDA_VISIBLE_DEVICES=0 $PY_CMD tools/train.py --cfg experiments/cityscapes/seg_hrnet_baseline_full.yaml

