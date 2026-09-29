CUDA_VISIBLE_DEVICES=0
DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.1
NBITS=2
DEVICE=3

python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgWSQV1_"$ALPHA"_"B$NBITS" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
quantizer=WSQ quantizer.wt_bit=${NBITS} quantizer.wt_clip_prob=-1 \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS project="dev_quant3" \
# split.mode=iid
# quantizer=WSQ quantizer.wt_bit=${NBITS} \
# python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgWSQ_"$ALPHA"_"B$NBITS" \
