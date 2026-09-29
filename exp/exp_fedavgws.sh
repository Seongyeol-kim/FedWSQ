DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.03
DEVICE=0

python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgWS_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS project="ICCV_REBUTTAL" \
# split.mode=iid
                               