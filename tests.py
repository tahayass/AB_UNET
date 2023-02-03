import numpy as np 
import matplotlib.pyplot as plt
import os
from scipy import interpolate













'''
path=r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\experiments\high_dropout_value_with_addeed_epochs\dice_stats'
i=0
path2=r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\experiments\high_dropout_value_with_addeed_epochs2\dice_stats'
for np_files in os.listdir(path):
    colors=['r','b','g','y','c']
    offset=[2,4,6,8,10]
    dice_stats=np.load(os.path.join(path,np_files))
    dice_stats2=np.load(os.path.join(path2,np_files))
    x=np.arange(1,dice_stats.size+1,1)/dice_stats.size
    y=(dice_stats+dice_stats2)/2
    #y=dice_stats
    e=(abs(dice_stats2-dice_stats)/2)
    #plt.plot(x,y,label=np_files.replace('.npy',''),color=colors[i])
    plt.errorbar(x, y, yerr=e,capsize=1,errorevery=(offset[i],10),elinewidth=0.5,ecolor=colors[i],color=colors[i],label=np_files.replace('.npy',''))
    i=i+1
plt.plot(x,np.ones(dice_stats.size)*0.71,label='regular model training with full dataset')
plt.title("Test dice")
plt.ylim()
plt.xlabel("Dataset size")
plt.ylabel("Dice metric")
plt.legend()
plt.show()
'''
