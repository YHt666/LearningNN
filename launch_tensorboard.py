""" 启动tensorboard并每隔一段时间重新加载logs """
import subprocess
import os
import time


reload_interval = 30   # 重新加载的时间间隔(秒)
while True:
    p1 = subprocess.Popen('tensorboard --logdir=runs --port=6006 --reload_interval=5 --reload_multifile=true', 
                          shell=True)
    time.sleep(reload_interval)

    p2 = subprocess.Popen('netstat -ano|findstr "6006"', shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           text=True
                           )
    out, err = p2.communicate()
    tb_pid = out.split()[-1]
    os.system(f'taskkill /f /pid {tb_pid}')

    p1.terminate()
    p2.terminate()
    # time.sleep(reload_interval * 0.5)
