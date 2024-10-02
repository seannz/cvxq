#!/bin/bash -x

class="facebook"
group_size=512

valid_size=64
stride=256
# bitrate=3.00
pca=1024
gpus=2
max_iters=512
checkpointing="" #--checkpointing"

for model in opt-1.3b # opt-1.3b # opt-1.3b
do
    for bitrate in 3.0 4.0
    do
	for batch_size in 2 4 8 32  # 8 32 # 16 8 4 2 # 8 4 2 1
	do
	    remarks="$class"-"$model"_group_size_"$group_size"_batch_size_"$batch_size"_bit_rate_"$bitrate"_stride_"$stride"_postsearch
	    python opt_train.py $checkpointing --model_id $class/$model --group_size $group_size --batch_size $batch_size --valid_size $valid_size --bitrate $bitrate --train_tokens 128 --max_iters $max_iters --stride $stride --pca $pca --gpus $gpus | tee -a results/batch_size_postsearch/$remarks.txt
	done
    done
done
