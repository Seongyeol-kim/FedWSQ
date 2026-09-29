@echo off

cd ..\

set DATASETS=cifar10 cifar100
set BITS=1 2 4 8
set METHODS=PAQ HQ
set ALPHA=0.3
set PARTICIPATION_RATE=0.02

set NUM_CLIENTS=500
set BATCH_SIZE=10

if "%DATASETS%" == "tinyimagenet" (
    set BATCH_SIZE=20
)

for %%D in (%DATASETS%) do (
    for %%M in (%METHODS%) do (
        for %%B in (%BITS%) do (
            echo Running experiment with DATASET=%%D, METHOD=%%M, BITS=%%B, NUM_CLIENTS=%NUM_CLIENTS%
            python federated_train.py client=base server=base quantizer=%%M ^
            exp_name=Fed%%M_%%D_B%%B_C%NUM_CLIENTS% dataset=%%D trainer.num_clients=%NUM_CLIENTS% ^
            split.alpha=%ALPHA% trainer.participation_rate=%PARTICIPATION_RATE% ^
            batch_size=%BATCH_SIZE% wandb=True model=resnet18_WS quantizer.wt_bit=%%B ^
            project="FedAVG_WS_Quant"
        )
    )
)
