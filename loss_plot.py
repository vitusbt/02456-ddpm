import numpy as np
import matplotlib.pyplot as plt

"""
A script for loading the loss data and plotting it
"""

run_name = 'mnist_final1'

loss = np.load('epoch_loss_'+run_name+'.npy')

plt.figure(dpi=300)
plt.semilogy(np.arange(1,len(loss)+1), loss)
#plt.plot(np.arange(1,len(loss)+1), loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.grid()
plt.subplots_adjust(left=0.15, right=0.925)
plt.savefig('loss_plot_'+run_name+'.png')
plt.show()