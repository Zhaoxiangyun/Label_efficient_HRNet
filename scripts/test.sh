 CUDA_VISIBLE_DEVICES=1 python tools/test.py --cfg experiments/cityscapes/baseline_10.yaml \
                          TEST.MODEL_FILE  output/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/epoch_239_check_state.pth \
                          TEST.FLIP_TEST True
