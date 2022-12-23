from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from torchnet.dataset import SplitDataset

from datasets import load_from_disk
import requests
from tqdm.auto import tqdm
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForTokenClassification, BertTokenizerFast, EvalPrediction, T5Tokenizer
from transformers import TrainingArguments
from transformers import Trainer
import itertools


model_name = "Rostlab/prot_t5_xl_uniref50"
max_length = 128

unique_tags = []
tag2id = {}
id2tag = {}

# Create dataset
class CreateDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def convert_to_char_list(input_list):
    char_list = []
    for s in input_list:
        char_list.append(list(s))
    return char_list

# Mask disorder tokens
def mask_disorder(labels, masks):
    for label, mask in zip(labels,masks):
        for i, disorder in enumerate(mask):
            if disorder == "0.0":
                #shift by one because of the CLS token at index 0
                label[i+1] = -100

def encode_tags(tags, encodings):    
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    pad_token = -100
    encoded_labels = zip(*itertools.zip_longest(*labels, fillvalue=pad_token))
    return list(encoded_labels)

def prepare_dataset(data_dir):
    dataset = load_from_disk(data_dir)
    # dataset = load_dataset("agemagician/NetSurfP-SS3")
    train_dataset = dataset.data['train']
    test_dataset = dataset.data['test']
    validation_dataset = dataset.data['validation']
    
    test_seq = test_dataset['input'].to_pylist()
    test_labels = test_dataset['label'].to_pylist()
    test_disorder = test_dataset['disorder'].to_pylist()
    
    val_seq = validation_dataset['input'].to_pylist()
    val_labels = validation_dataset['label'].to_pylist()
    val_disorder = validation_dataset['disorder'].to_pylist()
    
    train_seq = train_dataset['input'].to_pylist()
    train_labels = train_dataset['label'].to_pylist()
    train_disorder = train_dataset['disorder'].to_pylist()
    
    train_seq = convert_to_char_list(train_seq)
    val_seq = convert_to_char_list(val_seq)
    test_seq = convert_to_char_list(test_seq)
    
    train_labels = convert_to_char_list(train_labels)
    val_labels = convert_to_char_list(val_labels)
    test_labels = convert_to_char_list(test_labels)
    
    train_label_1 = [ list(label)[:max_length] for label in train_labels]
    val_label_1 = [list(label)[:max_length] for label in val_labels]
    test_label_1 = [ list(label)[:max_length] for label in test_labels]
    
    seq_tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    
    train_seq_encodings = seq_tokenizer(train_seq, is_split_into_words=True, 
                                    # return_offsets_mapping=True, 
                                    truncation=True, 
                                    padding=True, 
                                    max_length = 1024)
    val_seq_encodings = seq_tokenizer(val_seq, is_split_into_words=True, 
                                      # return_offsets_mapping=True, 
                                      truncation=True, 
                                      padding=True, 
                                      max_length = 1024)
    test_seq_encodings = seq_tokenizer(val_seq, 
                                       is_split_into_words=True, 
                                       # return_offsets_mapping=True, 
                                       truncation=True, 
                                       padding=True, 
                                       max_length = 1024)
    
    # tokenize labels
    # Consider each label as a tag for each token
    global unique_tags
    global tag2id
    global id2tag
    
    unique_tags = set(tag for doc in train_labels for tag in doc)
    unique_tags  = sorted(list(unique_tags))  # make the order of the labels unchanged
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    train_labels_encodings = encode_tags(train_label_1, train_seq_encodings)
    val_labels_encodings = encode_tags(val_label_1, val_seq_encodings)
    test_labels_encodings = encode_tags(test_label_1, test_seq_encodings)
    
    mask_disorder(train_labels_encodings, train_disorder)
    mask_disorder(val_labels_encodings, val_disorder)
    mask_disorder(test_labels_encodings, test_disorder)
    
    train_dataset = CreateDataset(train_seq_encodings, train_labels_encodings)
    val_dataset = CreateDataset(val_seq_encodings, val_labels_encodings)
    test_dataset = CreateDataset(test_seq_encodings, test_labels_encodings)
    
    return train_dataset, val_dataset, test_dataset


def model_init():
    return T5ForConditionalGeneration.from_pretrained(model_name,
                                                         num_labels=len(unique_tags),
                                                         id2label=id2tag,
                                                         label2id=tag2id,
                                                         )
def main():
    parser = argparse.ArgumentParser()

    # parsing the hyperparameters
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--subset', type=int, default=250)
    parser.add_argument('--num_workers', type=int, default=4) # for multi-process loading 
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_batch_size', type=int, default=8)

    # PyTorch container environment variables for data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_DATA'])
    # parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument("--save-model", action="store_true", default=True, help="For Saving the current Model")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="For displaying smdistributed.dataparallel-specific logs",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args, _ = parser.parse_known_args()
        
    train_dataset, val_dataset, test_dataset  = prepare_dataset(args.data)
 
    training_args = TrainingArguments(
    output_dir=args.model_dir,          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    learning_rate=3e-05,             # learning rate
    weight_decay=0.0,                # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    do_predict=True,
    evaluation_strategy="steps",     # evalute after every eval_steps
    eval_steps=100,
    gradient_accumulation_steps=32,  # total number of steps before back propagation
    fp16=True,                       # Use mixed precision
    fp16_opt_level="02",             # mixed precision mode
    run_name="ProBert-T5-XL",      # experiment name
    seed=3,                         # Seed for experiment reproducibility
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="steps",
    max_grad_norm=0,
    dataloader_drop_last=True,
    )
    
    trainer = Trainer(
    model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    )

    trainer.train()
    
    trainer.save_model(args.model_dir)

if __name__ =='__main__':
    main()
