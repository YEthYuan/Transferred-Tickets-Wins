import os

pydir = r"/home/yuanye/anaconda3/envs/t38/bin/python"

# for pr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
for pr in [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
    shell_code = f"{pydir} image_inversion.py --prune_rate {pr}"
    print("Ready to execute: ", shell_code)
    os.system(shell_code)
