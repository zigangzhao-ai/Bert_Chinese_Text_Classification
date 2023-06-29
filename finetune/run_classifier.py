"""
This script provides an example to wrap UER-py for classification.
"""
import sys
import os
import random
import argparse
import torch
import torch.nn as nn
import csv

import torch.distributed as dist

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.utils.misc import pooling
from uer.utils.callbacks import LossHistory
from uer.model_saver import save_model
from uer.opts import finetune_opts, tokenizer_opts, adv_opts


class Classifier(nn.Module):
    def __init__(self, args):

        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling_type = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


class ClassifierInfer(nn.Module):
    def __init__(self, args):

        super(ClassifierInfer, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling_type = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        return logits


def count_labels_num(prefix_pth, paths):
    labels_set, columns = set(), {}
    # with open(path, mode="r", encoding="utf-8") as f:
    paths = paths.split(',')
    print('===', paths)
    for path in paths:
        data_pth = os.path.join(prefix_pth, path)
        with open(data_pth, 'r') as tsvfile:
            f = csv.reader(tsvfile, delimiter='\t')
            for line_id, line in enumerate(f):
                if line_id == 0:
                    # for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    for i, column_name in enumerate(line):
                        columns[column_name] = i  #columns={'label': 0, 'text_a': 1}
                    continue
            
                txt = line[columns["text_a"]] #.rstrip("\r\n").split("\t")
                label = int(line[columns["label"]])
                labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"), strict=False)
        # model.output_layer_2 = nn.Linear(768, 2)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    elif args.optimizer in ["adafactor"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                scale_parameter=False, relative_step=False)
    elif args.optimizer in ["lion"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate)

    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, paths):
    dataset, columns = [], {}
    # with open(path, mode="r", encoding="utf-8") as f:
    paths = paths.split(',')
    for path in paths:
        data_pth = os.path.join(args.data_pth, path)
        with open(data_pth, 'r') as tsvfile:
            f = csv.reader(tsvfile, delimiter='\t')
            for line_id, line in enumerate(f):
                if line_id == 0:
                    # for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    for i, column_name in enumerate(line):
                        columns[column_name] = i  #columns={'label': 0, 'text_a': 1}
                        # import pdb; pdb.set_trace()
                    continue
                # line = line.rstrip("\r\n").split("\t")
                # text_a = "\t".join(line[1:])
                # src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                # txt = line[1].rstrip("\r\n").split("\t")
                # tgt = int(new_line[columns["label"]])
                # tgt = int(line[columns["label"]])
                tgt = int(line[columns["label"]])
                assert tgt >= 0, "gt_label is error, it is must >= 0"
                
                # print('====', tgt)
                if args.soft_targets and "logits" in columns.keys():
                    soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
                if "text_b" not in columns:  # Sentence classification.
                    # text_a = new_line[columns["text_a"]]
                    # text_a = line[columns["text_a"]].rstrip("\r\n").replace(' ', '\t').split("\t")
                    # text_a = line[columns["text_a"]].rstrip("\r\n").split("\t")
                    text_a = [line[columns["text_a"]].rstrip("\r\n").replace('\t', ' ')]
                    # print('-----+++++', len(text_a))
                    out_line = []
                    for split_txt in text_a:
                        new_split_txt = args.tokenizer.tokenize(split_txt)
                        out_line +=  new_split_txt + [SEP_TOKEN]
                    # import pdb; pdb.set_trace()    
                    # src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                    src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + out_line)
                    # print('----',[CLS_TOKEN], [SEP_TOKEN], src)
                    # import pdb; pdb.set_trace()   
                    seg = [1] * len(src)
                else:  # Sentence-pair classification.
                    text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                    src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                    src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                    src = src_a + src_b
                    seg = [1] * len(src_a) + [2] * len(src_b)

                if len(src) > args.seq_length:
                    src = src[: args.seq_length]
                    seg = seg[: args.seq_length]
                PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
                while len(src) < args.seq_length:
                    src.append(PAD_ID)
                    seg.append(0)
                if args.soft_targets and "logits" in columns.keys():
                    dataset.append((src, tgt, seg, soft_tgt))
                else:
                    dataset.append((src, tgt, seg))
    # if shuffle:
    # random.shuffle(dataset)
    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    # print('++++', loss)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
    # print('++++----', loss)
    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    if args.use_adv and args.adv_type == "fgm":
        args.adv_method.attack(epsilon=args.fgm_epsilon)
        loss_adv, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
        if torch.cuda.device_count() > 1:
            loss_adv = torch.mean(loss_adv)
        loss_adv.backward()
        args.adv_method.restore()

    if args.use_adv and args.adv_type == "pgd":
        K = args.pgd_k
        args.adv_method.backup_grad()
        for t in range(K):
            # apply the perturbation to embedding
            args.adv_method.attack(epsilon=args.pgd_epsilon, alpha=args.pgd_alpha,
                                   is_first_attack=(t==0))
            if t != K - 1:
                model.zero_grad()
            else:
                args.adv_method.restore_grad()
            loss_adv, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            if torch.cuda.device_count() > 1:
                loss_adv = torch.mean(loss_adv)
            loss_adv.backward()
        args.adv_method.restore()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        # print('---', gold)
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    args.logger.info("Confusion matrix:")
    args.logger.info(confusion)
    args.logger.info("Report precision, recall, and f1:")

    eps = 1e-9
    confusion_result = {}
   
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        args.logger.info("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
        confusion_result[i] = {}
        confusion_result[i]['p'] = p
        confusion_result[i]['r'] = r
        confusion_result[i]['f1'] = f1

    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion_result


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    parser.add_argument("--shuffle", type=bool, default=False,
                        help="Weight of the soft targets loss.")
    parser.add_argument("--log_dir", type=str, default="checkpoints/bert",
                        help="Weight of the soft targets loss.")
    parser.add_argument("--data_pth", type=str, default="datasets/boss_concact/high_q_0524",
                        help="the pth of data.")
    
    # parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
    #                     help='number of data loading workers (default: 4)')
    # parser.add_argument('-g', '--gpus', default=1, type=int,
    #                     help='number of gpus per node')
    # parser.add_argument('-nr', '--nr', default=0, type=int,
    #                     help='ranking within the nodes')
    
    adv_opts(parser)
    args = parser.parse_args()
    

    # os.environ['MASTER_ADDR'] = '127.0.0.1'   # 设置的是通讯的IP地址。在的单机单卡或者单机多卡中，可以设置为'127.0.0.1'（也就是本机）。在多机多卡中可以设置为结点0的IP地址
    # os.environ['MASTER_PORT'] = '8888'   # 设置通讯的端口，可以随机设置，只要是空闲端口就可以。
    # args.world_size = args.gpus * args.nodes
    # rank = args.nr * args.gpus + gpu
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    # Count the number of labels.
    args.labels_num = count_labels_num(args.data_pth, args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    args.logger = init_logger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(args.device)
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path) ##default = False
    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))
    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        
    args.model = model

    if args.use_adv:
        args.adv_method = str2adv[args.adv_type](model)

    total_loss, result, best_result = 0.0, 0.0, 0.0

    args.logger.info("Start training.")
    loss_history = LossHistory(args)
    
    total_step = 0
    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(trainset)
        src = torch.LongTensor([example[0] for example in trainset])
        tgt = torch.LongTensor([example[1] for example in trainset])
        seg = torch.LongTensor([example[2] for example in trainset])
        if args.soft_targets:
            soft_tgt = torch.FloatTensor([example[3] for example in trainset])
        else:
            soft_tgt = None

        model.train()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            total_step += 1
            if total_step % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, total_step, total_loss / args.report_steps))
                loss_history.append_loss(epoch, total_step, total_loss / args.report_steps)
                total_loss = 0.0
               
            if total_step % args.eval_steps == 0:
                result = evaluate(args, read_dataset(args, args.dev_path))
                loss_history.append_eval_result(epoch, total_step, result)

                if result[0] > best_result and epoch >= 2:
                    best_result = result[0]
                    os.makedirs(args.output_model_path, exist_ok=True)
                    save_model_name = args.output_model_path + '/'+ 'model'+'_epoch{}'.format(epoch)+'_step{}'.format(total_step)+'_acc_{}'.format(round(result[0],4)) +'.bin'
                    save_model(model, save_model_name)

    last_save_model_name = args.output_model_path + '/' + 'model' + '_last' + '.bin'
    save_model(model, last_save_model_name)

    #Evaluation phase.
    if args.test_path is not None:
        args.logger.info("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(last_save_model_name))
        else:
            args.model.load_state_dict(torch.load(last_save_model_name))
        evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()
