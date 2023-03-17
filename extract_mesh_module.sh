DATA=scannet_scene0_add_v3_errormask # scannet_scene0_origin_v4
NEAR=5
FAR=20

# SEQ=${DATA}_near${NEAR}_far${FAR} 
SEQ=scannet_scene0_add_v3_errormask_near5_far20
# SEQ=scannet_scene0_add_v3_finetune_on_origin_v4_pointmask_near5_far20_tea1024_wtea0.0001
CUDA_VISIBLE_DEVICES=1 python extract_mesh.py \
        --config configs/$DATA.txt \
        --expname $SEQ # \
        # --near $NEAR --far $FAR

# CUDA_VISIBLE_DEVICES=1 python extract_mesh.py --config configs/lego.txt
