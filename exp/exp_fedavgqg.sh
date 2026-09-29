DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 

ALPHA=0.6
NBITS=2
DEVICE=3


python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgQGV1_iid_"B$NBITS" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
quantizer=WSQG quantizer.wt_bit=${NBITS} quantizer.momentum=0.1 quantizer.wt_clip_prob=-1 \
batch_size=${BATCH_SIZE} wandb=True project="dev_quant3" \
split.mode=iid
# quantizer=WSQ quantizer.wt_bit=${NBITS} \d
# python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgWSQ_"$ALPHA"_"B$NBITS" \
