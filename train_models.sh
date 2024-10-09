#!/bin/bash

# Loop over the seed values
for seed in 1 2 3 4 5
do
  # Train with the default initialization
  python train.py .\CIFAR\cifar10 --dataset torch/cifar10 --model efficientnetv2_s --num-classes 10 --input-size 3 32 32 --batch-size 128 --amp --no-ddp-bb --opt adamw --weight-decay 0.1 --clip-grad 1.5 --sched-on-updates --lr 0.001 --min-lr 1e-6 --epochs 3 --warmup-epochs 1 --warmup-lr 1e-6 --aa v0 --seed $seed --log-wandb --project-wandb COMPARE_INIT --run-wandb DEFAULT_INIT_$seed --model-kwargs initialization=goog

  # Train with the custom initialization
  python train.py .\CIFAR\cifar10 --dataset torch/cifar10 --model efficientnetv2_s --num-classes 10 --input-size 3 32 32 --batch-size 128 --amp --no-ddp-bb --opt adamw --weight-decay 0.1 --clip-grad 1.5 --sched-on-updates --lr 0.001 --min-lr 1e-6 --epochs 3 --warmup-epochs 1 --warmup-lr 1e-6 --aa v0 --seed $seed --log-wandb --project-wandb COMPARE_INIT --run-wandb CUSTOM_INIT_$seed --model-kwargs initialization=custom
done