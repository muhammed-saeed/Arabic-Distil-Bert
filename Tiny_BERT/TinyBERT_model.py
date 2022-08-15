
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import logging
logging.basicConfig(level=logging.ERROR)
from transformers import BertConfig



output_model_file = '/local/home/CE/musaeed/TinyBERT/pytorch_model.bin'
output_vocab_file = '/local/home/CE/musaeed/TinyBERT/'




config = BertConfig.from_pretrained(
    "asafaya/bert-base-arabic"
)

model = BertModel.from_pretrained("asafaya/bert-base-arabic")
tokenizer = BertTokenizer.from_pretrained("asafaya/bert-base-arabic")

config.to_json_file('/local/home/CE/musaeed/TinyBERT/config.json')
model_to_save = model
torch.save(model_to_save.state_dict(), output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')
print('This tutorial is completed')