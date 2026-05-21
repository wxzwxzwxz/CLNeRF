###  
CONFIG=kitchen
DATA=Kitchen
### 

###  
CUDA=1
OP=DEL
###

EXP=${DATA}_${OP}_clnerf
FT=${DATA} 
rm -r logs/$EXP 

CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
    --config configs/CLNeRF/$CONFIG.txt --expname $EXP --datadir ./data/$DATA \
    --transforms_train single_operation/${OP}/transforms_train_newview.json \
    --transforms_val single_operation/${OP}/transforms_test_newview.json \
    --transforms_test single_operation/${OP}/transforms_test_newview.json \
    --ft_path logs/$FT/200000.tar --N_iters 210000 --i_video 220000 --i_weight 10000 --i_testset 1000 \
    --use_teacher_nerf --datadir_teacher data/$DATA \
    --transforms_train_teacher single_operation/${OP}/transforms_train_oldview.json \
    --ft_teacher_path logs/$FT/200000.tar 


for FACTOR in 8 0
do
    for ITER in {210..210..1}
    do
        CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
            --config configs/CLNeRF/$CONFIG.txt \
            --datadir data/$DATA \
            --expname $EXP --render_only --render_test \
            --render_factor $FACTOR --testskip 0 \
            --transforms_train single_operation/${OP}/transforms_test_newview.json \
            --transforms_val single_operation/${OP}/transforms_test_newview.json \
            --transforms_test single_operation/${OP}/transforms_test_newview.json --render_output_name renderonly_test_stage1_newtask \
            --ckpt_path logs/${EXP}/${ITER}000.tar 

        # OLD
        CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
            --config configs/CLNeRF/$CONFIG.txt \
            --datadir data/$DATA \
            --expname $EXP --render_only --render_test \
            --render_factor $FACTOR --testskip 0 \
            --transforms_train single_operation/${OP}/transforms_test_oldview.json \
            --transforms_val single_operation/${OP}/transforms_test_oldview.json \
            --transforms_test single_operation/${OP}/transforms_test_oldview.json --render_output_name renderonly_test_stage1_oldtask \
            --ckpt_path logs/${EXP}/${ITER}000.tar 
    done
done
