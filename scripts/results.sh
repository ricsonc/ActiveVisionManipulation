#!/bin/bash

function run() {
    OPENBLAS_NUM_THREADS=1 python3.6 ../HER/examples/visualize.py --render-eval "$@"
}

#uncomment one of the below experiments

#active abstr:
# run --env-id=rcnn_pusher_1_2_0-v0 --factor=1 \
#     --log-dir=../test_results/active_abstr_test \
#     --restore-dir=../saved_weights/active_abstr

#static:
# run --env-id=rcnn_pusher_1_3_0-v0 --factor=1 \
#     --log-dir=../test_results/static_test \
#     --restore-dir=../saved_weights/static

#active abstr reward
# run --env-id=rcnn_pusher_1_2_0-v0 --factor=1 \
#     --log-dir=../test_results/active_abstr_visrwd_test \
#     --restore-dir=../saved_weights/active_abstr_visrwd

#active full
# run --env-id=rcnn_pusher_1_2_0-v0 --factor=0 \
#     --log-dir=../test_results/active_full \
#     --restore-dir=../saved_weights/active_full

#random camera movements
# run --env-id=rcnn_randpusher_1_2_0-v0 --factor=1 \
#     --log-dir=../test_results/randcam_test \
#     --restore-dir=../saved_weights/randcam
