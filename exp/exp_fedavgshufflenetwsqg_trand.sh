DATASET=cifar100
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3
DEVICE=1

python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgShuffleNetWSQGTRand_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
quantizer=WSQG quantizer.random_bit=rand_alloc quantizer.momentum=0.1 quantizer.wt_clip_prob=-1 \
batch_size=${BATCH_SIZE} wandb=True model=ShuffleNet_WS project="dev_quant3" \

# split.mode=iid
# quantizer=WSQ quantizer.wt_bit=${NBITS}
# python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgWSQ_"$ALPHA"_"B$NBITS" \
