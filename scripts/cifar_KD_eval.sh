#!/usr/bin/env bash

gpus=${1}
model_name=${2}
dataset_name=${3}
distillation=${4}
bit=${5}
optimize=${6}
lambda=${7}

home=$PWD
scope=Student


if [[ ${dataset_name} == cifar10 ]]; then
    data_dir=${home}/datasets/cifar-10-batches-py
elif [[ ${dataset_name} == cifar100 ]]; then
    data_dir=${home}/datasets/cifar-100-python
fi

bit_a=${bit}
bit_w=${bit}
bit_g=32

lr=0.001
wd=1e-05
epochs=200
bs=128


search_folder=${dataset_name}_${model_name}_${scope}_${bit_a}_${wd}_${epochs}_${lr}_${bs}_${distillation}_${optimize}_${lambda}_KD_search
search_dir=${home}/results/${search_folder}


# the best policy id for ResNet18 + CIFAR100 with 2-bit activations and weights.
idx=0

eval_folder=${dataset_name}_${model_name}_${scope}_${bit_a}_${wd}_${epochs}_${lr}_${bs}_${distillation}_${optimize}_${lambda}_${idx}_KD_eval
eval_dir=${home}/results/${eval_folder}
if [[ ! -d "${eval_dir}" ]]; then
    echo 'mkdir eval dir'
    mkdir ${eval_dir}
fi

CUDA_VISIBLE_DEVICES=${gpus} python pba/train.py --local_dir ${home}/results/ --data_path ${data_dir} --dataset ${dataset_name} \
--model_name ${model_name} --train_size 50000 --val_size 0 --checkpoint_freq 20 \
--name "${eval_folder}" \
--gpu 1 --cpu 6 --num_samples 1 --hp_policy_epochs ${epochs} --epochs ${epochs} --explore cifar10 --aug_policy cifar10 \
--lr ${lr} --wd ${wd} --main_scope ${scope} --use_hp_policy --opt ${optimize} --lamb ${lambda} \
--hp_policy ${search_dir}/pbt_policy_${idx}.txt \
--bit_a ${bit_a} --bit_w ${bit_w} --bit_g ${bit_g} \
--load ${home}/results/${dataset_name}_${model_name}_Teacher_AUG/model.ckpt-200 \
--Distillation ${distillation} \
--teacher ${home}/results/models/${dataset_name}_${model_name}_Teacher_AUG.mat
