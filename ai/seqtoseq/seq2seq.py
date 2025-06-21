# import os
# import torch
# import math
# import random
import spacy
# going from one sequence to another
# rnn to encode the input sequence into single vector
# this vector is then decoded by a second rnn, 
# which learns to output the target sequence one at a time
# input: embedding of current word + hidden state of previous time step
# output: new hidden state (vector representation of the sequence so far)
# ENGLISH TO SANSKRIT

# import torch
# import torch.nn as nn
# import torch.optim as optim
from datasets import load_dataset
from indicnlp.tokenize import indic_tokenize

SEED = 42

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

dataset = load_dataset("rahular/itihasa")
# print(dataset)

train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)

# print(dataset['train'][0])

# spacy tokenizers
# for english: python -m spacy download en_core_web_sm
en_nlp = spacy.load("en_core_web_sm")

english_str = "What a lovely day it is today!"
sanskrit_str = "अद्य किं मनोहरः दिवसः अस्ति!"

# print([token.text for token in en_nlp.tokenizer(string)])
# print('Input String: {}'.format(sanskrit_str))
# print('Tokens: ')
# for t in indic_tokenize.trivial_tokenize(sanskrit_str): 
#     print(t)
