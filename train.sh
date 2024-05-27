#!/bin/bash

# Batch size is lower than default 128 because, depending on the size of the dataset, 128 may exceed the total size of the dataset.

python train.py \
--batch-size 32 \
--model resnet18 \
--pretrained \
--num-classes 37 \
--opt adam \
--lr 3e-4 \
--data-dir data \
--epochs 50 \
--seed 0 \