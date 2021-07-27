PY_CMD="python -m torch.distributed.launch --nproc_per_node=2"

CUDA_VISIBLE_DEVICES=4,5 $PY_CMD tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml

