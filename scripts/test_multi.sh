 CUDA_VISIBLE_DEVICES=3 python tools/test.py --cfg experiments/cityscapes/seg_hrnet_baseline_0_05.yaml \
                          TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                          TEST.MODEL_FILE  output/cityscapes/seg_hrnet_second_0/check_state.pth \
                          TEST.FLIP_TEST True
