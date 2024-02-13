@echo off

:start
tensorboard --logdir=runs
choice /t 30 /d y /n >nul    %每隔n秒运行一次%

goto start