#!/bin/bash -x

class="meta-llama"
model="Llama-2-7b-hf"
group_size=256
batch_size=1
valid_size=1
stride=256
bitrate=2.1
pca=1024
gpus=1
max_iters=256
checkpointing="" #--checkpointing"

remarks="$class"-"$model"_group_size_"$group_size"_batch_size_"$batch_size"_bit_rate_"$bitrate"_pca_"$pca"_stride_"$stride"_search_single_channel_search_weighted_mult_bias_search_debug
python opt_train.py $checkpointing --model_id $class/$model --group_size $group_size --batch_size $batch_size --valid_size $valid_size --bitrate $bitrate --train_tokens 128 --max_iters $max_iters --stride $stride --pca $pca --gpus $gpus | tee -a results/$remarks.txt
