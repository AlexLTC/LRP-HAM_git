import os, sys

os.system('/home/user/xuus_blue/LRP-HAM/experiments/scripts/train_P4.sh 0 throat_uvula res101')
os.system('/home/user/xuus_blue/LRP-HAM/experiments/scripts/train_LRP_HAI_P4.sh 0 throat_uvula res101 40000 110000 True')
os.system('/home/user/xuus_blue/LRP-HAM/experiments/scripts/test_LRP_HAI_P4.sh 0 throat_uvula res101 0 True')
# os.system('python /home/user/xuus_blue/LRP-HAM/tools/demo_drl.py')
# os.system('python /home/user/xuus_blue/LRP-HAM/tools/demo_drl.py')
