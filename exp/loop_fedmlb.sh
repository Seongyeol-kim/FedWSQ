#!/bin/bash

data_sets=(cifar10 cifar100 tinyimagenet)
alpha_values=(0.05 0.1 0.3 0.6)
SEED=100
DEVICE=0

# Iterate over datasets
for DATASET in "${data_sets[@]}"; do
    # Set default BATCH_SIZE based on dataset
    if [ "$DATASET" = "tinyimagenet" ]; then
        BATCH_SIZE=100
    else
        BATCH_SIZE=50
    fi

    # Iterate over split modes
    for SPLIT_MODE in "iid" "dirichlet"; do

        if [ "$SPLIT_MODE" = "iid" ]; then
            # For iid mode, no need to iterate over alpha
            ALPHA=0.6
            EXP_NAME="FedMLB_iid"
            python3 federated_train.py client=MLB server=base visible_devices=\'$DEVICE\' seed=$SEED \
                exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                batch_size="$BATCH_SIZE" wandb=True model=resnet18_MLB project="FedWS_5_100_seed2"
        else
            # For non-iid mode, iterate over alpha values
            for ALPHA in "${alpha_values[@]}"; do
                EXP_NAME=FedMLB_"$ALPHA"
                python3 federated_train.py client=MLB server=base visible_devices=\'$DEVICE\' seed=$SEED \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                    batch_size="$BATCH_SIZE" wandb=True model=resnet18_MLB project="FedWS_5_100_seed2"
            done
        fi
    done
done
