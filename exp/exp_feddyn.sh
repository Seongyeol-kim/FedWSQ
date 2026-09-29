DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3
DEVICE=1

python3 federated_train.py client=Dyn server=FedDyn visible_devices=\'$DEVICE\' exp_name=FedDyn_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True project="FedWS_5_100" save_freq=500 \
# split.mode=iid
                                                                                                  