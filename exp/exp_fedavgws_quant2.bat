@echo off

cd ..\

set DATASETS=tinyimagenet
set BITS=1 2 4 8
set METHODS=PAQ HQ
set ALPHA=0.3
set PARTICIPATION_RATE=0.02

set NUM_CLIENTS=100
set BATCH_SIZE=50

if "%DATASETS%" == "tinyimagenet" (
    set BATCH_SIZE=100
)

for %%D in (%DATASETS%) do (

    for %%M in (%METHODS%) do (
        for %%B in (%BITS%) do (
            for %%N in (%NUM_CLIENTS%) do (
                echo Running experiment with DATASET=%%D, METHOD=%%M, BITS=%%B, NUM_CLIENTS=%%N
                python federated_train.py client=base server=base quantizer=%%M ^
                exp_name=Fed%%M_%%D_B%%B_C%%N dataset=%%D trainer.num_clients=%%N ^
                split.alpha=%ALPHA% trainer.participation_rate=%PARTICIPATION_RATE% ^
                batch_size=%BATCH_SIZE% wandb=True model=resnet18_WS quantizer.wt_bit=%%B ^
                project="FedAVG_WS_Quant"
            )
        )
    )
)
