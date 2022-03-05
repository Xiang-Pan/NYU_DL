'''
Author: Xiang Pan
Date: 2022-03-04 16:28:14
LastEditTime: 2022-03-04 20:26:27
LastEditors: Xiang Pan
Description: 
FilePath: /HW2/hw2_rnn_part2.py
@email: xiangpan@nyu.edu
'''
import torch
import random
import string

import torch

# Max value of the generated integer. 26 is chosen becuase it's
# the number of letters in English alphabet.
N = 26


def idx_to_onehot(x, k=N+1):
    """ Converts the generated integers to one-hot vectors """
    ones = torch.sparse.torch.eye(k).cuda()
    shape = x.shape
    res = ones.index_select(0, x.view(-1).type(torch.int64).cuda())
    return res.view(*shape, res.shape[-1])


def s_to_idx(s):
    """ Converts a string to a list of integers """
    return [(ord(c) - ord('a') + 1) for c in s]

def idx_to_s(idx):
    # a is class 1, b is class 2
    """ Converts a list of integers to a string """
    return ''.join([chr(c + ord('a') - 1) for c in idx])

class VariableDelayEchoDataset(torch.utils.data.IterableDataset):

    def __init__(self, max_delay=8, seq_length=20, size=1000):
        self.max_delay = max_delay
        self.seq_length = seq_length
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        for _ in range(self.size):
            seq = torch.tensor([random.choice(range(1, N + 1)) for i in range(self.seq_length)], dtype=torch.int64)
            delay = random.randint(0, self.max_delay)
            result = torch.cat((torch.zeros(delay), seq[:self.seq_length - delay])).type(torch.int64)
            yield seq, delay, result

def test_variable_delay_model(model, seq_length=20):
    """
    This is the test function that runs 100 different strings through your model,
    and checks the error rate.
    """
    total = 0
    correct = 0
    for i in range(500):
        s = ''.join([random.choice(string.ascii_lowercase) for i in range(seq_length)])
        d = random.randint(0, model.max_delay)
        result = model.test_run(s, d)
        if d > 0:
            z = zip(s[:-d], result[d:])
        else:
            z = zip(s, result)
        for c1, c2 in z:
            correct += int(c1 == c2)
        total += len(s) - d

    return correct / total


import time
start_time = time.time()

MAX_DELAY = 8
SEQ_LENGTH = 20

# TODO: implement model training here.
model = VariableDelayGRUMemory(hidden_size=512, max_delay=MAX_DELAY)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300], gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=4, verbose=True)
criterion = torch.nn.CrossEntropyLoss()

dataset = VariableDelayEchoDataset(max_delay=8, seq_length=20, size=1000)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

epoch = 0
test_acc = 0
while test_acc <= 0.99:
    epoch_loss = 0
    for x, d, y in train_dataloader:
        model.train()
        optimizer.zero_grad()
        d = d.unsqueeze(1)
        logits = model(x, d)
        loss = criterion(logits.reshape(-1, N + 1), y.reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_dataloader)
    # scheduler.step(epoch_loss)
    test_acc = test_variable_delay_model(model)
    print(f'epoch {epoch}: {test_acc}', epoch_loss)
    epoch += 1

end_time = time.time()
assert end_time - start_time < 600, 'executing took longer than 10 mins'
assert test_variable_delay_model(model) > 0.99, 'accuracy is too low'
print('tests passed')