CUDA=5
CONFIG=kitchen
DATA=Kitchen
OP=ADD

for FACTOR in 8 0
do
    for ITER in {210..210..1}
    do
        for EXP in Kitchen_${OP}_clnerf
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
done