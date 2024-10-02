#!/bin/bash -x

class="meta-llama"
model="opt-350m"
group_size=256
valid_size=64
stride=128
# bitrate=3.00
pca=1024
gpus=4
max_iters=256 #1024
checkpointing="--checkpointing"

for model in Llama-2-70b-hf # opt-350m opt-1.3b opt-2.7b opt-6.7b opt-13b opt-30b #Llama-2-7b-hf #256 # 512 Llama-2-13b-hf # opt-13b # opt-1.3b
do
    for bitrate in 4.0 3.0 #3.0 4.0 # 4.0 # 2.1 2.2 2.4 2.6 2.8 # 3.0 3.2 3.4 3.6 3.8 4.0 #4.0 3.0
    do
	for batch_size in 16 # 16 8 4 2 # 8 4 2 1
	do
	    remarks="$class"-"$model"_group_size_"$group_size"_batch_size_"$batch_size"_bit_rate_"$bitrate"_stride_"$stride"_max_iters_"$max_iters"_debug_final #_checkpointing
	    python opt_train.py $checkpointing --model_id $class/$model --group_size $group_size --batch_size $batch_size --valid_size $valid_size --bitrate $bitrate --train_tokens 128 --max_iters $max_iters --stride $stride --pca $pca --gpus $gpus --remarks $remarks | tee -a results/opt_all/$remarks.txt
	done
    done
done
