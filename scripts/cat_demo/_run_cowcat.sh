#!/usr/bin/env bash
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
export CUDA_VISIBLE_DEVICES=2,7
export VIDEO_DIR=video_dir/cowcat_demo
export LOG=outputs/cowcat_demo.log
export TB_DIR=tb/cowcat_demo
export VLFM_OBJECTNAV_QUERY=找奶牛猫
export VLFM_ATTR_NOUN=cat
export VLFM_ATTR_PREDICATE="a black-and-white cow-patterned cat"
export VLFM_ATTR_VERIFY=1
export VLFM_ATTR_FAIL_OPEN=0
export ATTR_VERIFIER_PORT=12186
exec bash scripts/cat_demo/eval_cat_demo.sh
