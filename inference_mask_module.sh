DATA=blendswap_whitehouse_origin_v7
NEAR=1
FAR=10
EXP=${DATA}_near${NEAR}_far${FAR}

CUDA_VISIBLE_DEVICES=5 python run_nerf.py \
    --config configs/$DATA.txt \
    --datadir data/blendswap_whitehouse_origin_v9 \
    --near $NEAR --far $FAR --expname $EXP --render_only --render_test --render_mask_only \
    --testskip 0 --render_factor 0 