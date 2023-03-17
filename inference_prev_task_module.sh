DATA=scannet_scene0_add_v3
# DATA=scannet_scene0_origin_v4
# DATA=scannet_scene0_add_v3_errormask
NEAR=5
FAR=20
# EXP=${DATA}_near${NEAR}_far${FAR}
# EXP=scannet_scene0_add_v3_near${NEAR}_far${FAR}
# EXP=scannet_scene0_add_v3_finetune_on_origin_v4_near5_far20_tea1024_wtea0.0001

# for EXP in scannet_scene0_add_v3_finetune_on_origin_v4_near5_far20_tea1024_wtea0.0001 \
# scannet_scene0_add_v3_finetune_on_origin_v4_near5_far20_tea1024_wtea0.001 \
# scannet_scene0_add_v3_finetune_on_origin_v4_near5_far20_tea1024_wtea0.01 \
# scannet_scene0_add_v3_finetune_on_origin_v4_near5_far20_tea1024_wtea0.1 \
# scannet_scene0_add_v3_finetune_on_origin_v4_pointmask_near5_far20_tea1024_wtea0.00001 \
# scannet_scene0_add_v3_finetune_on_origin_v4_pointmask_near5_far20_tea1024_wtea0.0001 \
# scannet_scene0_add_v3_finetune_on_origin_v4_pointmask_near5_far20_tea512_wtea0.0001 \
# scannet_scene0_add_v3_finetune_on_origin_v4_pointmask_near5_far20_tea512_wtea0.001

for EXP in scannet_scene0_add_v3_finetune_on_origin_v4_pointmask_near5_far20_tea1024_wtea0.00001
do      
    CUDA_VISIBLE_DEVICES=2 python run_nerf.py \
        --config configs/$DATA.txt \
        --datadir data/scannet_scene0_origin_v4 \
        --near $NEAR --far $FAR --expname $EXP --render_only --render_test \
        --render_wo_images --ori_H 1080 --ori_W 1920 \
        --testskip 0 --render_factor 0 # --chunk 1024
done