@echo off

cd ..\

set CUDA_VISIBLE_DEVICES=0

set DATASET=cifar10
set BATCH_SIZE=50
if "%DATASET%" == "tinyimagenet" (
    set BATCH_SIZE=100
)

set PARTICIPATION_RATE=0.05
set QUANTIZERS=PAQ

for %%A in (0.05 0.1 0.3 0.6 1) do (
    for %%W in (1 2 4 8) do (
        echo Running experiment with ALPHA=%%A and WT_BIT=%%W
        python federated_train.py client=base server=base quantizer=%QUANTIZERS% ^
        exp_name=Fed%QUANTIZERS%_%%A_B%%W dataset=%DATASET% trainer.num_clients=100 ^
        split.alpha=%%A trainer.participation_rate=%PARTICIPATION_RATE% ^
        batch_size=%BATCH_SIZE% wandb=True model=resnet18_WS quantizer.wt_bit=%%W ^
        project="FedAVG_WS_Quant"
    )
)
