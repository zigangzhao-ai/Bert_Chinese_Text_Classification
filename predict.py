"""
  This script provides an example to  UER-py for classification inference - single sentence.
"""
import sys
import os
import torch
import argparse
import collections
import torch.nn as nn
import logging
import csv
import xlsxwriter as xw

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.opts import infer_opts, tokenizer_opts, model_opts
from finetune.run_classifier import Classifier

def load_model(model, model_path):
    """
    Load model from saved weights.
    """
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model

def bert_pred(args, model, chat_text, device):

    chat_text = chat_text.rstrip("\r\n").replace('\t', ' ')
    token_txt = [CLS_TOKEN] + args.tokenizer.tokenize(chat_text) + [SEP_TOKEN]
    src = args.tokenizer.convert_tokens_to_ids(token_txt)
    seg = [1] * len(src)

    if len(src) > args.seq_length:
        src = src[: args.seq_length]
        seg = seg[: args.seq_length]
    # PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
    # while len(src) < args.seq_length:
    #     src.append(PAD_ID)
    #     seg.append(0)
    # print('++++', token_txt, src, seg)

    src_batch = torch.LongTensor([src]).to(device)
    seg_batch = torch.LongTensor([seg]).to(device)
    print('===', src_batch.shape, seg_batch.shape)
    
    with torch.no_grad():
        _, logits = model(src_batch, None, seg_batch)

    pred = torch.argmax(logits, dim=1)
    pred = pred.cpu().numpy().tolist()[0]
    
    probs = nn.Softmax(dim=1)(logits)
    probs = probs.cpu().numpy().tolist()[0][pred]
    print('---', logits, pred, probs)
    return pred, probs

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path options.
    parser.add_argument("--load_model_path", type=str, default="/code/zzg/project/NLP/code/UER-py/checkpoints/roberta/model_0629.bin",
                        help="Path of the input model.")
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json",
                        help="Path of the config file.")
    parser.add_argument("--seq_length", type=int, default=512, #512
                        help="Sequence length.")
    parser.add_argument("--labels_num", type=int, default=2,
                        help="Number of prediction labels.")
    # Model options.
    model_opts(parser)
    tokenizer_opts(parser)

    args = parser.parse_args()
    args.vocab_path = "models/google_zh_vocab.txt"
    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    print('---', args)
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
    model.eval()
  
    chat_text = "我的电话号码是 135 6783 4356" 
    pred, _ = bert_pred(args, model, chat_text, device)
    print(pred)
            
if __name__ == "__main__":
    main()
