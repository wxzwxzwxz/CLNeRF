CUDA=5
CONFIG=kitchen
DATA=Kitchen
EXP=${DATA}_SEQ_clnerf

rm -rf logs/${EXP}
mkdir logs/${EXP}
cp logs/${DATA}_ADD_clnerf/210000*.tar logs/${EXP}

OP=DEL
FT=${EXP}
CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
    --config configs/CLNeRF/$CONFIG.txt --expname $EXP --datadir ./data/$DATA \
    --transforms_train single_operation/${OP}/transforms_train_newview.json \
    --transforms_val single_operation/${OP}/transforms_test_newview.json \
    --transforms_test single_operation/${OP}/transforms_test_oldview.json \
    --testskip 0 \
    --ft_path logs/$FT/210000.tar --N_iters 220000 --i_video 300000 --i_weight 10000 --i_testset 1000 \
    --use_teacher_nerf --datadir_teacher data/${DATA} \
    --transforms_train_teacher single_operation/${OP}/transforms_train_oldview.json \
    --ft_teacher_path logs/$FT/210000.tar \
    --use_teacher_nerf_with_branch --expert_teacher_num 1

OP=MOV
FT=${EXP}
CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
    --config configs/CLNeRF/$CONFIG.txt --expname $EXP --datadir ./data/$DATA \
    --transforms_train single_operation/${OP}/transforms_train_newview.json \
    --transforms_val single_operation/${OP}/transforms_test_newview.json \
    --transforms_test single_operation/${OP}/transforms_test_oldview.json \
    --testskip 0 \
    --ft_path logs/$FT/220000.tar --N_iters 230000 --i_video 300000 --i_weight 10000 --i_testset 1000 \
    --use_teacher_nerf --datadir_teacher data/$DATA \
    --transforms_train_teacher single_operation/${OP}/transforms_train_oldview.json \
    --ft_teacher_path logs/$FT/220000.tar \
    --use_teacher_nerf_with_branch --expert_teacher_num 2

OP=REP
FT=${EXP}
CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
    --config configs/CLNeRF/$CONFIG.txt --expname $EXP --datadir ./data/$DATA \
    --transforms_train sequential_operation/transforms_train_newview.json \
    --transforms_val sequential_operation/transforms_test_newview.json \
    --transforms_test sequential_operation/transforms_test_oldview.json \
    --testskip 0 \
    --ft_path logs/$FT/230000.tar --N_iters 240000 --i_video 300000 --i_weight 10000 --i_testset 1000 \
    --use_teacher_nerf --datadir_teacher data/$DATA \
    --transforms_train_teacher single_operation/${OP}/transforms_train_oldview.json \
    --ft_teacher_path logs/$FT/230000.tar \
    --use_teacher_nerf_with_branch --expert_teacher_num 3

./experiments/inference/inference_kitchen_after_SEQ_clnerf.sh