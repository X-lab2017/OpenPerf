# import csv
import logging

# make deterministic
# from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig

# from mingpt.utils import sample
import random
import torch
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--context_length", type=int, default=30)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--model_type", type=str, default="naive")
parser.add_argument("--num_steps", type=int, default=500000)
parser.add_argument("--num_buffers", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--data_dir_prefix",
    type=str,
    default="./github_dataset/min_freq_160_preprocessed_data.pkl",
)
args = parser.parse_args()

# set_seed(args.seed)


class GithubDataset(Dataset):

    def __init__(self, data, labels, idx, mask):
        self.data = data
        self.labels = labels
        self.idx = idx
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index], dtype=torch.int64),
            self.labels[index],
            self.idx[index],
            self.mask[index],
            # torch.tensor(self.labels[index], dtype=torch.int64),
            # torch.tensor(self.idx[index], dtype=torch.int64),
        )


def load_dataset():
    data_path = args.data_dir_prefix
    with open(data_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)

    return data


def sequential_dataset(data, labels, max_len=30):
    seq = {}
    seq_data, seq_labels, seq_idx, seq_mask = [], [], [], []
    for i in range(len(data)):
        if data[i] not in seq:
            seq[data[i]] = [labels[i]]
        else:
            seq[data[i]].append(labels[i])

    for k, v in seq.items():
        split_data(k, v, seq_data, seq_labels, seq_idx, seq_mask, max_len)
        # v = torch.tensor(np.array(v)).long()
    seq_data = np.array(seq_data).reshape(-1, 1)
    # seq_labels = np.array(seq_labels)
    # seq_idx = np.array(seq_idx)

    return seq_data, seq_labels, seq_idx, seq_mask


def split_data(data, labels, data_list, label_list, idx_list, mask_list, max_len=30):
    if len(labels) == max_len:
        data_list.append(data)
        label_list.append(torch.tensor(np.array(labels), dtype=torch.int64))
        idx_list.append(
            torch.tensor(np.array([i for i in range(len(labels))]), dtype=torch.int64)
        )
        mask_list.append(
            torch.tensor(np.array([1 for _ in range(max_len)]), dtype=torch.int64)
        )
    elif len(labels) < max_len:
        data_list.append(data)
        mask_tmp = [1 for _ in range(len(labels))]
        idx_tmp = [i for i in range(len(labels))]
        for _ in range(max_len - len(labels)):
            labels.append(0)
            mask_tmp.append(0)
            idx_tmp.append(len(labels) - 1)
        label_list.append(torch.tensor(np.array(labels), dtype=torch.int64))
        idx_list.append(torch.tensor(np.array(idx_tmp), dtype=torch.int64))
        mask_list.append(torch.tensor(np.array(mask_tmp), dtype=torch.int64))
    else:
        data_list.append(data)
        label_list.append(torch.tensor(np.array(labels[:max_len]), dtype=torch.int64))
        idx_list.append(
            torch.tensor(np.array([i for i in range(max_len)]), dtype=torch.int64)
        )
        mask_list.append(
            torch.tensor(np.array([1 for _ in range(max_len)]), dtype=torch.int64)
        )
        split_data(
            data,
            labels[max_len:],
            data_list,
            label_list,
            idx_list,
            mask_list,
        )


def send_to_device(model):
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        # model.to(device)
        model = torch.nn.DataParallel(model).to(device)


data = load_dataset()
train_data, valid_data, test_data = data["train"], data["valid"], data["test"]
user_size, item_size = train_data.num_users, train_data.num_items
train_data, train_labels, train_idx, train_mask = sequential_dataset(
    train_data.user_ids, train_data.item_ids, args.context_length
)
# valid_data, valid_labels, valid_idx = sequential_dataset(
#     valid_data.user_ids, valid_data.item_ids, args.context_length+1
# )
test_data, test_labels, test_idx, test_mask = sequential_dataset(
    test_data.user_ids, test_data.item_ids, args.context_length
)


# set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

train_dataset = GithubDataset(train_data, train_labels, train_idx, train_mask)
# valid_dataset = GithubDataset(valid_data, valid_labels, valid_idx)
test_dataset = GithubDataset(test_data, test_labels, test_idx, test_mask)

mconf = GPTConfig(
    item_size,
    args.context_length * 2,
    n_layer=6,
    n_head=8,
    n_embd=128,
    model_type=args.model_type,
    max_timestep=args.context_length,
    user_num=user_size,
)
model = GPT(mconf)
send_to_device(model)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(
    max_epochs=epochs,
    batch_size=args.batch_size,
    learning_rate=6e-4,
    lr_decay=True,
    warmup_tokens=512 * 20,
    final_tokens=1 * len(train_dataset) * args.context_length * 2,
    num_workers=4,
    seed=args.seed,
    model_type=args.model_type,
    max_timestep=args.context_length,
)
trainer = Trainer(model, train_dataset, test_dataset, tconf)

trainer.train()
