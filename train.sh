#!/bin/bash

# Batch size is lower than default 128 because, depending on the size of the dataset, 128 may exceed the total size of the dataset.

# NUM_CLASSES=37   # oxford pets
NUM_CLASSES=102  # flowers 102


python train.py \
--batch-size 32 \
--model resnet18 \
--pretrained \
--num-classes $NUM_CLASSES \
--opt adam \
--lr 3e-4 \
--data-dir data \
--epochs 50 \
--seed 0 \
