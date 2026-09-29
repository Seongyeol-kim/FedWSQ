#!/bin/bash

data_sets=(cifar10 cifar100 tinyimagenet)
alpha_values=(0.05 0.1 0.3 0.6)
seeds=(301 609)
DEVICE=0

# Iterate SEED
for SEED in "${seeds[@]}"; do
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

            # Skip already completed case: CIFAR-10 + IID + seed 301
            if [ "$SEED" = "301" ] && [ "$DATASET" = "cifar10" ] && [ "$SPLIT_MODE" = "iid" ]; then
                echo "Skipping completed case: FedProx / cifar10 / iid / seed=$SEED"
                continue
            fi

            if [ "$SPLIT_MODE" = "iid" ]; then
                # For iid mode, no need to iterate over alpha
                ALPHA=0.6
                EXP_NAME="FedProx_iid_${SEED}"
                python federated_train.py client=Prox server=base visible_devices=\'$DEVICE\' seed=$SEED \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                    batch_size="$BATCH_SIZE" wandb=True project="iotj"
            else
                # For non-iid mode, iterate over alpha values
                for ALPHA in "${alpha_values[@]}"; do
                    EXP_NAME="FedProx_${ALPHA}_${SEED}"
                    python federated_train.py client=Prox server=base visible_devices=\'$DEVICE\' seed=$SEED \
                        exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                        split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                        batch_size="$BATCH_SIZE" wandb=True project="iotj"
                done
            fi
        done
    done
done