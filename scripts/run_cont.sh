PY_CMD="python -m torch.distributed.launch --nproc_per_node=1"

CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/cityscapes/seg_hrnet_contrastive_0_05.yaml

