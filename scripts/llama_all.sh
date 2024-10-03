#!/bin/bash -x

class="meta-llama"
group_size=256
valid_size=64
stride=128
# bitrate=3.00
pca=1024
gpus=1
max_iters=256 #1024
checkpointing="--checkpointing"

for model in Llama-2-70b-hf # Llama-2-70b-hf
do
    for bitrate in 3.0 4.0
    do
	for batch_size in 16
	do
	    remarks="$class"-"$model"_group_size_"$group_size"_batch_size_"$batch_size"_bit_rate_"$bitrate"_stride_"$stride"_max_iters_"$max_iters"_timing #_checkpointing
	    python opt_train.py $checkpointing --model_id $class/$model --group_size $group_size --batch_size $batch_size --valid_size $valid_size --bitrate $bitrate --train_tokens 128 --max_iters $max_iters --stride $stride --pca $pca --gpus $gpus --remarks $remarks | tee -a results/llama_all/$remarks.txt
	done
    done
done
