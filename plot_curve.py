import os
import matplotlib.pyplot as plt

def loss_plot(loss_seq,lr_seq,  path = 'Train_hist.png', model_name = ''):
    x = range(len(loss_seq))

    y1 = loss_seq
    y2 = lr_seq

    plt.plot(x, y1, label='loss')
    plt.plot(x, y2, label='lr')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)

    #plt.close()
    print("finish")
    plt.show()
    plt.close()

if __name__ == '__main__':
    log_file = "./snapshots/voc12/log.txt"
    log_data_list = [item.strip() for item in open(log_file, 'r')]
    length = len(log_data_list)
    print("the number of records:", length)

    loss_seq =[]
    lr_seq =[]
    for item in log_data_list:
        print( item.split())
        if len(item.split())==5 :
            if item.split()[3]=="lr:":
                _, _, loss_val,_, lr = item.split()
                loss_val = float(loss_val)
                lr = float(lr)
                loss_seq.append(loss_val)
                lr_seq.append(lr)
                print("loss_val:", loss_val)
                print("lr:", lr)
    loss_plot(loss_seq, lr_seq, path = 'Train_hist.png', model_name = '')
