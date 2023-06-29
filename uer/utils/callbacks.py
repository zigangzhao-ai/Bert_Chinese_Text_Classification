import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class LossHistory():
    def __init__(self, args):
      
        self.log_dir = args.log_dir  #os.path.join(log_dir, "weights_" + str(time_str))
        self.losses  = []
        self.val_loss = [] 
        self.time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
   
        # self.writer = SummaryWriter(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.f1 = open(os.path.join(self.log_dir, "epoch_train_loss.txt"), 'a')
        self.f1.write('Start_time:' + ' ' + self.time_str + '\n')
        self.f1.write('Config_path:' + ' ' + args.config_path + '\n')
        self.f1.write('Train_path:' + ' ' + args.train_path + '\n')
        self.f1.write('Dev_path:' + ' ' + args.dev_path + '\n')
        self.f1.write('Epochs_num:' + ' ' + str(args.epochs_num) + '\n')
        self.f1.write('Seq_length:'+ ' ' + str(args.seq_length) + '\n')
        self.f1.write('Batch_size:' + ' ' + str(args.batch_size) + '\n')
        self.f1.write('Learning_rate:' + ' ' + str(args.learning_rate) + '\n')
        self.f1.write('Scheduler:' + ' ' + str(args.scheduler) + '\n') #optimizer
        self.f1.write('Optimizer:' + ' ' + str(args.optimizer) + '\n')
        # self.f1 = open(os.path.join(self.log_dir, "epoch_train_loss.txt"), 'a')
           
    def append_loss(self, epoch=None, total_step=None, train_loss=None, train_acc=None, val_loss=None, val_acc=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(train_loss)
        # self.val_loss.append(val_loss)
        line = 'epoch: ' + str(epoch) + ' || ' + 'step: ' + str(total_step) + ' || '+ 'train_loss: ' + str(train_loss) 
        self.f1.write(line + '\n')
        # with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
        #     line = 'epoch: ' + str(epoch) + ' || ' + 'val_loss: ' +  str(val_loss) + ' || ' + 'val_acc: ' + str(val_acc)
        #     f.write(line + '\n')
        # self.writer.add_scalar('train_loss', train_loss, epoch)
        # self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()
    
    def append_eval_result(self, epoch, total_step, eval_result):

        eval_acc, confusion_result = eval_result
        line1 = 'epoch: ' + str(epoch) + ' || ' + 'eval_step:' + str(total_step) + ' || ''eval_acc: ' + str(eval_acc)
        self.f1.write(line1 + '\n')

        for key in confusion_result:
            line1 = '\t' +'Label: ' + str(key) + '  ' + str(confusion_result[key]) 
            self.f1.write(line1 + '\n')
       
    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        # plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            # plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Iiter/20')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")
