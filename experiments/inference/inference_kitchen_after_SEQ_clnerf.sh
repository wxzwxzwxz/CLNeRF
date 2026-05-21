CUDA=5
CONFIG=kitchen
DATA=Kitchen
OP=SEQ

for FACTOR in 8 0
do
    for ITER in {240..240..1}
    do
        for EXP in ${DATA}_${OP}_clnerf
        do
            # NEW
            CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
                --config configs/CLNeRF/$CONFIG.txt \
                --datadir data/$DATA \
                --expname $EXP --render_only --render_test \
                --render_factor $FACTOR --testskip 0 \
                --transforms_train sequential_operation/transforms_test_newview.json \
                --transforms_val sequential_operation/transforms_test_newview.json \
                --transforms_test sequential_operation/transforms_test_newview.json --render_output_name renderonly_test_stage1_newtask \
                --ckpt_path logs/${EXP}/${ITER}000.tar \
                --use_teacher_nerf_with_branch --ft_teacher_path logs/${EXP}/${ITER}000.tar --expert_teacher_num 3

            # OLD
            CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
                --config configs/CLNeRF/$CONFIG.txt \
                --datadir data/$DATA \
                --expname $EXP --render_only --render_test \
                --render_factor $FACTOR --testskip 0 \
                --transforms_train sequential_operation/transforms_test_oldview.json \
                --transforms_val sequential_operation/transforms_test_oldview.json \
                --transforms_test sequential_operation/transforms_test_oldview.json --render_output_name renderonly_test_stage1_oldtask \
                --ckpt_path logs/${EXP}/${ITER}000.tar \
                --use_teacher_nerf_with_branch --ft_teacher_path logs/${EXP}/${ITER}000.tar --expert_teacher_num 3
            
            # OLD
            CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
                --config configs/CLNeRF/$CONFIG.txt \
                --datadir data/$DATA \
                --expname $EXP --render_only --render_test \
                --render_factor $FACTOR --testskip 0 \
                --transforms_train sequential_operation/transforms_test_oldview_step0_ADD.json \
                --transforms_val sequential_operation/transforms_test_oldview_step0_ADD.json \
                --transforms_test sequential_operation/transforms_test_oldview_step0_ADD.json --render_output_name renderonly_test_stage1_oldtask_step0_ADD \
                --ckpt_path logs/${EXP}/${ITER}000.tar \
                --use_teacher_nerf_with_branch --ft_teacher_path logs/${EXP}/${ITER}000.tar --expert_teacher_num 3
            
            # OLD
            CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
                --config configs/CLNeRF/$CONFIG.txt \
                --datadir data/$DATA \
                --expname $EXP --render_only --render_test \
                --render_factor $FACTOR --testskip 0 \
                --transforms_train sequential_operation/transforms_test_oldview_step1_DEL.json \
                --transforms_val sequential_operation/transforms_test_oldview_step1_DEL.json \
                --transforms_test sequential_operation/transforms_test_oldview_step1_DEL.json --render_output_name renderonly_test_stage1_oldtask_step1_DEL \
                --ckpt_path logs/${EXP}/${ITER}000.tar \
                --use_teacher_nerf_with_branch --ft_teacher_path logs/${EXP}/${ITER}000.tar --expert_teacher_num 3
            
            # OLD
            CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
                --config configs/CLNeRF/$CONFIG.txt \
                --datadir data/$DATA \
                --expname $EXP --render_only --render_test \
                --render_factor $FACTOR --testskip 0 \
                --transforms_train sequential_operation/transforms_test_oldview_step2_MOV.json \
                --transforms_val sequential_operation/transforms_test_oldview_step2_MOV.json \
                --transforms_test sequential_operation/transforms_test_oldview_step2_MOV.json --render_output_name renderonly_test_stage1_oldtask_step2_MOV \
                --ckpt_path logs/${EXP}/${ITER}000.tar \
                --use_teacher_nerf_with_branch --ft_teacher_path logs/${EXP}/${ITER}000.tar --expert_teacher_num 3
        done
    done
done