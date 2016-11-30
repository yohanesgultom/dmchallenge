'''
Visualize loss and accuracy from Keras training log in simple plots
'''
import re
import sys
import os
import matplotlib.pyplot as plt

# pattern = r'.* loss: (\d+\.[-e\d\+]*) - acc: (\d+\.[-e\d\+]*) - true_pos: (\d+\.[-e\d\+]*) - true_neg: (\d+\.[-e\d\+]*) - val_loss: (\d+\.[-e\d\+]*) - val_acc: (\d+\.[-e\d\+]*) - val_true_pos: (\d+\.[-e\d\+]*) - val_true_neg: (\d+\.[-e\d\+]*).*'
pattern = r'.* loss: (\d+\.[-e\d\+]*) - acc: (\d+\.[-e\d\+]*) - true_pos: (\d+\.[-e\d\+]*) - true_neg: (\d+\.[-e\d\+]*).*'

if __name__ == '__main__':
    logfilepath = sys.argv[1]
    loss = []
    acc = []
    true_pos = []
    true_neg = []
    val_loss = []
    val_acc = []
    val_true_pos = []
    val_true_neg = []
    with open(logfilepath, 'r') as logfile:
        for line in logfile:
            epoch_end = re.match(pattern, line, re.M | re.I)
            if epoch_end:
                loss.append(float(epoch_end.group(1)))
                acc.append(float(epoch_end.group(2)))
                true_pos.append(float(epoch_end.group(3)))
                true_neg.append(float(epoch_end.group(4)))
                # val_loss.append(float(epoch_end.group(5)))
                # val_acc.append(float(epoch_end.group(6)))
                # val_true_pos.append(float(epoch_end.group(7)))
                # val_true_neg.append(float(epoch_end.group(8)))

    print(len(loss))

    filename = os.path.basename(logfilepath)

    f1_title = filename + ' loss & accuracy'
    f1 = plt.figure(f1_title)
    ax1 = f1.add_subplot(111)
    ax1.plot(loss)
    ax1.plot(acc)
    ax1.set_xlabel('Epoch')
    ax1.set_title(f1_title)
    ax1.legend(['loss', 'acc'], loc='upper right')

    # f2_title = filename + ' validation loss & accuracy'
    # f2 = plt.figure(f2_title)
    # ax2 = f2.add_subplot(111)
    # ax2.plot(val_loss)
    # ax2.plot(val_acc)
    # ax2.set_xlabel('Epoch')
    # ax2.set_title(f2_title)
    # ax2.legend(['val_loss', 'val_acc'], loc='upper right')

    f3_title = filename + ' true positive & true negative'
    f3 = plt.figure(f3_title)
    ax3 = f3.add_subplot(111)
    ax3.plot(true_pos)
    ax3.plot(true_neg)
    ax3.set_xlabel('Epoch')
    ax3.set_title(f3_title)
    ax3.legend(['true_pos', 'true_neg'], loc='upper right')

    # f4_title = filename + ' validation true positive & true negative'
    # f4 = plt.figure(f4_title)
    # ax4 = f4.add_subplot(111)
    # ax4.plot(val_true_pos)
    # ax4.plot(val_true_neg)
    # ax4.set_xlabel('Epoch')
    # ax4.set_title(f4_title)
    # ax4.legend(['val_true_pos', 'val_true_neg'], loc='upper right')

    plt.show()
