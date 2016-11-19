'''
Visualize loss and accuracy from Keras training log in simple plots
'''
import re
import sys
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    logfilepath = sys.argv[1]
    loss = []
    acc = []
    val_loss = []
    val_acc = []
    with open(logfilepath, 'r') as logfile:
        for line in logfile:
            epoch_end = re.match(r'.* loss: (\d+\.[-e\d]*) - acc: (\d+\.[-e\d]*) - val_loss: (\d+\.[-e\d]*) - val_acc: (\d+\.[-e\d]*).*', line, re.M | re.I)
            if epoch_end:
                loss.append(float(epoch_end.group(1)))
                acc.append(float(epoch_end.group(2)))
                val_loss.append(float(epoch_end.group(3)))
                val_acc.append(float(epoch_end.group(4)))

    print(len(loss))

    filename = os.path.basename(logfilepath)
    f1_title = filename + ' loss & accuracy'
    f1 = plt.figure(f1_title)
    ax1 = f1.add_subplot(111)
    ax1.plot(loss)
    ax1.plot(acc)
    ax1.set_xlabel('Batch')
    ax1.set_title(f1_title)
    ax1.legend(['loss', 'acc'], loc='upper right')

    f2_title = filename + ' validation loss & accuracy'
    f2 = plt.figure(f2_title)
    ax2 = f2.add_subplot(111)
    ax2.plot(val_loss)
    ax2.plot(val_acc)
    ax2.set_xlabel('Epoch')
    ax2.set_title(f2_title)
    ax2.legend(['val_loss', 'val_acc'], loc='upper right')
    plt.show()
