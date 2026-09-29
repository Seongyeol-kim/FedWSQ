CUDA_VISIBLE_DEVICES=0
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3
DEVICE=2

python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgMobileViTWS_$ALPHA \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True model=MobileViTWS project="FedWS_5_100" \
# split.mode=iid
