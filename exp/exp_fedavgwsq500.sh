CUDA_VISIBLE_DEVICES=0
DATASET=cifar10
BATCH_SIZE=10
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=20
fi 
ALPHA=0.3
NBITS=2

python3 federated_train.py client=base server=base visible_devices=\'0\' exp_name=FedAvgWSQ_"$ALPHA"_"B$NBITS" \
dataset=${DATASET} trainer.num_clients=500 split.alpha=${ALPHA} trainer.participation_rate=0.02 \
quantizer=WSQ quantizer.wt_bit=${NBITS} \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS project="dev_quant2" \
# split.mode=iid

