DEVICE=1
DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.03

python3 federated_train.py client=fedrcl server=base visible_devices=\'$DEVICE\' exp_name=FedRCL_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True project="ICCV_REBUTTAL" \
# split.mode=iid &
