CUDA=0
CONFIG=kitchen
DATA=Kitchen

for FACTOR in 8 0
do
    for ITER in {200..200..1}
    do
        for EXP in Kitchen_near0.1_far10_ts2
        do
            # NEW
            CUDA_VISIBLE_DEVICES=$CUDA python run_nerf.py \
                --config configs/$CONFIG.txt \
                --datadir data/$DATA \
                --expname $EXP --render_only --render_test \
                --render_factor $FACTOR --testskip 2 \
                --transforms_train original/transforms_test.json \
                --transforms_val original/transforms_test.json \
                --transforms_test original/transforms_test.json --render_output_name renderonly_test_stage0 \
                --ckpt_path logs/${EXP}/${ITER}000.tar 

        done
    done
done