DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.6
DEVICE=1

python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgWSCutmixReg0.2_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 dataset.cutmix.use=true dataset.cutmix.cutmix_reg=true \
dataset.cutmix.prob=0.2 batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS project="cutmix" \
# split.mode=iid