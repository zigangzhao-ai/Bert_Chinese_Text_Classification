import onnxruntime
import argparse
import cv2
import sys
import os
import torch
import time
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)
print(sys.path)

from uer.utils import *
from uer.utils.constants import *
from uer.utils.config import load_hyperparam
from uer.opts import tokenizer_opts, model_opts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax(x): 
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Path options.

parser.add_argument("--max_seq_len", type=int, default=128, #512
                        help="Max sequence length.")  
parser.add_argument("--config_path", type=str, default="models/bert/base_config.json",
                    help="Path of the config file.") 
parser.add_argument("--onnx_path", type=str, default="",
                    help="Path of the onnx model.")                
# Model options.
model_opts(parser)
tokenizer_opts(parser)
args = parser.parse_args()
# Load the hyperparameters from the config file.
args.vocab_path = "models/google_zh_vocab.txt"
args = load_hyperparam(args)

# Build tokenizer.
args.tokenizer = str2tokenizer[args.tokenizer](args)

##tokens_to_ids
chat_text = "我的电话号码是 135 6783 4356" 
chat_text = chat_text.rstrip("\r\n").replace('\t', ' ')
token_txt = [CLS_TOKEN] + args.tokenizer.tokenize(chat_text) + [SEP_TOKEN]
src = args.tokenizer.convert_tokens_to_ids(token_txt)
seg = [1] * len(src)

if len(src) > args.max_seq_len:
    src = src[: args.max_seq_len]
    seg = seg[: args.max_seq_len]

src = np.array([src])
seg = np.array([seg])
# onnx_weights = "/code/zzg/project/NLP/code/UER-py/model_convert/onnx/bert_concact_own.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
session = onnxruntime.InferenceSession(args.onnx_path, providers=providers)
print(session.get_providers())
t1 = time.time()

input_name1 = session.get_inputs()[0].name
input_name2 = session.get_inputs()[1].name
print('----', input_name1, input_name2)

logits = session.run(None, {input_name1: src, input_name2: seg})[0][0]
pred = softmax(logits)
label = np.argmax(logits)
# print('----', pred, label)
conf = round(float(pred[label]), 5)
print('----', pred, label, conf)
t2 = time.time()
print(t2-t1)
