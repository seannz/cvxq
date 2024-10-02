#!/bin/bash -x

class="facebook"
model="opt-30b"
group_size=512
batch_size=16
valid_size=128
stride=128
bitrate=3.0
pca=1024
gpus=2
max_iters=1024
checkpointing="--checkpointing"

remarks="$class"-"$model"_group_size_"$group_size"_batch_size_"$batch_size"_bit_rate_"$bitrate"_pca_"$pca"_stride_"$stride"
python opt_train.py $checkpointing --model_id $class/$model --group_size $group_size --batch_size $batch_size --valid_size $valid_size --bitrate $bitrate --train_tokens 128 --max_iters $max_iters --stride $stride --pca $pca --gpus $gpus --remarks $remarks | tee -a results/postsearch/$remarks.txt
