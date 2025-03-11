#!/usr/bin/env bash

for count in 1
do
    echo "The value is: $count"
    CUDA_VISIBLE_DEVICES=0
    python train_mmoe_add.py \
    --seed $count \
    --lamb 0.2 \
    --x 4 \
    --model_name "HAIDC_$count.bin" \
    --batch_size 6 \
    --accumulation_steps 3 \
    --print_step 200 \
    --hidden_size 31 \
    --epoch 100 \
    --max_turns 5 \
    --max_seq_len 100 \
    --patience 3 \
    --coefficient 1 \
    --result_filename HAIDC_$count \
    --do_train 
done


