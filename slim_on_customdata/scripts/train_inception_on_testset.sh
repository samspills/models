#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifar_net_on_mnist.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/testset_training

# Where the dataset is saved to.
DATASET_DIR=/Users/work/Downloads/images_2016_08/train/test_output

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=testset \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception \
  --preprocessing_name=inception \
  --max_number_of_steps=100000 \
  --batch_size=128 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

# Run evaluation.
# python eval_image_classifier.py \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${TRAIN_DIR} \
#   --dataset_name=inception \
#   --dataset_split_name=validation \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=cifarnet
