import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer import TransformerModel
from data import MFCCDataset, load_data

import sys
import argparse

# Parameters
script_name = sys.argv[0]
parser = argparse.ArgumentParser(description="Perform classification on MFCCs and get the test accuracy.")
parser.add_argument("version", help="Dataset version", type=int, default=1)
parser.add_argument("data_dir", help="Data directory")
parser.add_argument("model_dir", help="Model directory")
parser.add_argument("n_class", help="Number of classes", type=int, default=12)
parser.add_argument('--low_res', help="Low resolution MFCC or not", action='store_true')
parser.add_argument("--use_model", help="Use trained model or not", action='store_true')
parser.add_argument("--subset", help="Use the subset of test set when testing", action='store_false')
args = parser.parse_args()

version = args.version
data_dir = args.data_dir
model_dir = args.model_dir
n_class = args.n_class
low_res = args.low_res
use_model = args.use_model
subset = args.subset

# Other parameters
nhead = 8
nhid = 256
nlayers = 4
dropout_pe = 0
dropout = 0

if low_res:
    n_feature = 13
    nembed = 32
else:
    n_feature = 40
    nembed = 64

epochs=100
patience=15
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if use_model:
    device = torch.device("cpu")

# Load data and get max sequnce length
train, validation, test, label2num = load_data(version, data_dir, n_class)
max_len = 0
def get_label(key):
    label = key.split("_")[-1]
    if label not in label2num:
        return 'other'
    return label
for k, val in train.items():
    seq_len = val.shape[0]
    if max_len < seq_len:
        max_len = seq_len
for _, val in validation.items():
    seq_len = val.shape[0]
    if max_len < seq_len:
        max_len = seq_len
for _, val in test.items():
    seq_len = val.shape[0]
    if max_len < seq_len:
        max_len = seq_len

num2label = {v: k for k, v in label2num.items()}

# Training
def run_one_epoch(train_flag, dataloader, model, optimizer, device):

    torch.set_grad_enabled(train_flag)
    model.train() if train_flag else model.eval() 

    losses = []
    accuracies = []

    for (x,y) in dataloader:
        (x, y) = (x.to(device).float(), y.to(device).float())

        output = model(x)
        loss = F.cross_entropy(output, y)
        if train_flag: 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        pred = torch.argmax(output, 1)
        true = torch.argmax(y, 1)
        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean((pred == true).float())
        accuracies.append(accuracy.detach().cpu().numpy()) 
    
    return(np.mean(losses), np.mean(accuracies))

def train_model(model, train_data, validation_data, device, epochs=100, patience=10, verbose = True):
    device = "cpu"
    model.to(device)

    train_dataset = MFCCDataset(train_data, max_len, n_feature, label2num, n_class)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = 0)
    validation_dataset = MFCCDataset(validation_data, max_len, n_feature, label2num, n_class)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, num_workers = 0)

    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt'
    train_accs = []
    val_accs = []
    for e in range(epochs):
        train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device)
        val_loss, val_acc = run_one_epoch(False, validation_dataloader, model, optimizer, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            patience_counter = patience
            torch.save(model.state_dict(), check_point_filename)
        else: 
            patience_counter -= 1
            if patience_counter <= 0: 
                model.load_state_dict(torch.load(check_point_filename))
                break
        if verbose:
                print("Epoch %i. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" % 
          (e+1, train_loss, train_acc, val_loss, val_acc, patience_counter ))

    return model, train_accs, val_accs


tf_model = TransformerModel(n_feature, n_class, nembed, nhead, nhid, nlayers, dropout, dropout_pe).to(device)
if use_model:
    tf_model.load_state_dict(torch.load(model_dir, map_location = device))
else:
    tf_model, train_accs, val_accs = train_model(tf_model, train, validation, device, epochs, patience)

# Test
test_dataset = MFCCDataset(test, max_len, n_feature, label2num, n_class)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers = 0)
device = 'cpu'
accuracies = []
with torch.no_grad():
    for (x,y) in test_dataloader:
        (x, y) = (x.to(device).float(), y.to(device).float())

        output = tf_model(x)
        pred = torch.argmax(output, 1)
        true = torch.argmax(y, 1)
        accuracy = torch.mean((pred == true).float())
        accuracies.append(accuracy)
	
	# Only run one batch 
        if subset:
                break

print("Test accuracy: ", (sum(accuracies) / len(accuracies)).item())
