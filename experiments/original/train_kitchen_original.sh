
CONFIG=kitchen
DATA=Kitchen
CUDA=0
EXP=${DATA}

CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
        --config configs/$CONFIG.txt --expname $EXP --datadir ./data/$DATA \
        --ext .jpg \
        --transforms_train original/transforms_train.json \
        --transforms_val original/transforms_test.json \
        --transforms_test original/transforms_test.json \
        --N_iters 200000 --i_video 210000 --i_test 10000 \
        --testskip 10