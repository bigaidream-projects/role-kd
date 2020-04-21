# Role-wise Data Augmentation for Knowledge Distillation
### Table of Contents

1. [Getting Started](#getting-started)
2. [Reproduce Results](#reproduce-results)
3. [Run Evaluation](#run-pba-search)
4. [Reference Code](#)

### Getting Started
Code supports Python 2.7 and will later support Python 3.6.

####  Install requirements

```
pip install -r requirements.txt 
```

#### Download CIFAR-10/CIFAR-100 datasets

```
bash datasets/cifar10.sh
bash datasets/cifar100.sh
```

### Reproduce Results
Scripts to reproduce results are located in `scripts/`. Currently, we only release an example for the inference stage ResNet18 with cifar100 using 2-bit weights and activations. And we will release the training codes when the paper is published. To reproduce the example result: 
```
bash scripts/cifar_KD_eval.sh ${gpu_id} ResNet18 cifar100 MHGD-RKD-SVD 2 adam 0.4
```

The result will be shown at 
```
results/cifar100_ResNet18_Student_2_1e-05_200_0.001_128_MHGD-RKD-SVD_adam_0.4_0_KD_eval/progress.csv
```

### Reference Code
- Augmentation policy
    - [Population Based Augmentation](https://github.com/arcelien/pba)
    - [Ray model](https://github.com/ray-project/ray/tree/master/python/ray/tune)
- Quantization
    - [Tensorpack](https://github.com/tensorpack/tensorpack)
- Knolwedge Distillation
    - [Knowledge distillation methods implemented with Tensorflow](https://github.com/sseung0703/Knowledge_distillation_methods_wtih_Tensorflow)
    
### Citation
```
@article{role-kd,
Author = {Jie Fu and Xue Geng and Zhijian Duan and Bohan Zhuang and Xingdi Yuan and Adam Trischler and Jie Lin and Chris Pal and Hao Dong},
Title = {Role-Wise Data Augmentation for Knowledge Distillation},
Year = {2020},
Eprint = {arXiv:2004.08861},
}

```
