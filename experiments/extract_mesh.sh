DATA=kitchen
EXP=Kitchen
CUDA_VISIBLE_DEVICES=2 python extract_mesh.py \
        --config configs/$DATA.txt \
        --expname $EXP 