"""
  This script provides an example to wrap UER-py for classification inference.
"""
import sys
import os
import torch
import argparse
import collections
import torch.nn as nn
import logging
import csv

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts
from finetune.run_classifier import Classifier


def batch_loader(batch_size, src, tgt, seg, text_a=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if text_a is not None:
            text_a_batch = text_a[i * batch_size : (i + 1) * batch_size]
        yield src_batch, tgt_batch, seg_batch, text_a_batch

    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if text_a is not None:
            text_a_batch = tgt[instances_num // batch_size * batch_size :]
        yield src_batch, tgt_batch, seg_batch, text_a_batch


def read_dataset(args, path):
    dataset, columns = [], {}
    # with open(path, mode="r", encoding="utf-8") as f:
    with open(path, 'r') as tsvfile:
        f = csv.reader(tsvfile, delimiter='\t')
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line):
                    columns[column_name] = i 
                continue
            # line = line.rstrip("\r\n").split("\t")
            # new_line = [line[0], line[1:]]
            if len(line[columns["label"]]) > 0:
                tgt = int(line[columns["label"]])
                #tgt = 0
            else:
                tgt = 0
            # tgt = int(line[columns["label"]])
            if "text_b" not in columns:  # Sentence classification.
                # text_a = new_line[columns["text_a"]]
                # text_a_out = "\t".join(text_a)
                # text_a = line[columns["text_a"]].rstrip("\r\n").split("\t")
                text_a = [line[columns["text_a"]].rstrip("\r\n").replace('\t', ' ')]
                text_a_out = "\t".join(text_a)
                # text_a = [line[columns["text_a"]].rstrip("\r\n").replace('\t', ' ')]  ##采用空格替代"\t", 效果可能更好
             
                text_a_out = line[columns["text_a"]].rstrip("\r\n")
                # print('-----+++++', text_a)
                out_line = []
                for split_txt in text_a:
                    new_split_txt = args.tokenizer.tokenize(split_txt)
                    out_line +=  new_split_txt + [SEP_TOKEN]
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
            dataset.append((src, tgt, seg, text_a_out))

    return dataset



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")

    tokenizer_opts(parser)

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # print('---', args.tokenizer)
    # import pdb; pdb.set_trace()

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    model = Classifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset = read_dataset(args, args.test_path)

    # print(dataset)
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    text_a = [sample[3] for sample in dataset]

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)
    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        
        f.write("gt" + '\t' + 'pred' + '\t' + 'prob' + '\t' +'text_a' + '\n')
        # f.write('pred'+'\n')
        correct = 0
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
        test_dataloader = batch_loader(batch_size, src, tgt, seg, text_a)
        for i, (src_batch, tgt_batch, seg_batch, text_a_batch) in enumerate(test_dataloader):
            
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            seg_batch = seg_batch.to(device)
           
            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)

            pred = torch.argmax(logits, dim=1)
            correct += torch.sum(pred == tgt_batch).item()

            for j in range(pred.size()[0]):
                confusion[pred[j], tgt_batch[j]] += 1
         
            pred = pred.cpu().numpy().tolist()
            gt = tgt_batch.cpu().numpy().tolist()

            probs = nn.Softmax(dim=1)(logits)
            probs = probs.cpu().numpy().tolist()
            new_probs = []
            for j, label in enumerate(pred):
                prob = probs[j][label]
                new_probs.append(round(prob, 4))
            
            for k in range(len(pred)):
                # if gt[k] != pred[k]:  
                if pred[k] in [1] and new_probs[k] < 0.85:
                #gt[k] != pred[k] and
                    txt = " ".join(str(text_a_batch[k]).split('\t'))
                    line = str(gt[k]) + '\t' + str(pred[k]) + '\t' + str(new_probs[k]) + '\t' + txt
                    # line =  str(pred[k]) + '\t' + str(new_probs[k]) + '\t' + txt
                    # line = str(pred[k])
                    print('+++', str(text_a_batch[k]))
                    f.write(line + '\n')
            # print('---pred--', pred, len(dataset))
            print('---correct_num---', correct)
            # print('---text---', text_a_batch)
            # print('---prob---', new_probs)
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 2 * p * r / (p + r + eps)
            print("Label {}: P {:.3f}, R {:.3f}, F1 {:.3f}".format(i, p, r, f1))

        ACC = correct / len(dataset)
        print('test_acc:', correct, len(dataset), round(ACC, 4))
        print(confusion)

if __name__ == "__main__":
    main()
