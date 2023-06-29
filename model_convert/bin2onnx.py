"""
functions: bert模型「.bin格式」转onnx

"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)
# print(sys.path)
from finetune.run_classifier import ClassifierInfer
from uer.utils import *
from uer.model_loader import load_model
from uer.utils.config import load_hyperparam
from uer.opts import tokenizer_opts, model_opts


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Path options.
parser.add_argument("--load_model_path", type=str, default="/code/zzg/project/NLP/code/UER-py/checkpoints/roberta/model_0629.bin",
                    help="Path of the input model.")
parser.add_argument("--config_path", type=str, default="models/bert/base_config.json",
                    help="Path of the config file.")
parser.add_argument("--max_seq_len", type=int, default=512, #512
                        help="Sequence length.")
parser.add_argument("--labels_num", type=int, default=2,
                    help="Number of prediction labels.")
parser.add_argument("--save_dir", type=str, default='',
                    help="Path of the save onnx.")
parser.add_argument("--onnx_name", type=str, default='',
                    help="Name of onnx.")

# Model options.
model_opts(parser)
tokenizer_opts(parser)

args = parser.parse_args()
# Load the hyperparameters from the config file.
args.vocab_path = "models/google_zh_vocab.txt"
args = load_hyperparam(args)

# Build tokenizer.
args.tokenizer = str2tokenizer[args.tokenizer](args)
print('---', args)

# Load model
model = ClassifierInfer(args)
model = load_model(model, args.load_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

#config
batch_size = 1
max_seq_len = args.max_seq_len

src = torch.randint(1, [batch_size, max_seq_len], dtype=torch.long).to(device)
seg = [1] * max_seq_len
seg = torch.LongTensor([seg]).to(device)
# print(src, seg)
print(src.shape, seg.shape)


os.makedirs(args.save_dir, exist_ok=True)
save_onnx_model = "{}/{}".format(args.save_dir, args.onnx_name)

example_input = (src, seg)
symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
# bin转onnx
with torch.no_grad():
    torch.onnx.export(model,
                      example_input,
                      save_onnx_model,
                      opset_version=13,
                      input_names=["input1", "input2"], 
                      output_names=["output"],
                      dynamic_axes={'input1': symbolic_names, 
                                    'input2': symbolic_names,             
                                    'output': symbolic_names}
                      )
print("-----convert finished !-------")





