'''
Author: Xiang Pan
Date: 2022-03-03 02:06:43
LastEditTime: 2022-03-04 21:52:04
LastEditors: Xiang Pan
Description: '''

import random
from re import S
import string

import torch
# import pytorch_lightning

# pytorch_lightning.seed_everything(2333)

# Max value of the generated integer. 26 is chosen becuase it's
# the number of letters in English alphabet.
N = 26


# N + 1 = 27 # from [0, 26] 26 is the blank


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

assert 'abcde' == idx_to_s(s_to_idx('abcde'))

class EchoDataset(torch.utils.data.IterableDataset):

    def __init__(self, delay=4, seq_length=15, size=1000):
        self.delay = delay
        self.seq_length = seq_length
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        """ Iterable dataset doesn't have to implement __getitem__.
            Instead, we only need to implement __iter__ to return
            an iterator (or generator).
        """
        for _ in range(self.size):
            seq = torch.tensor([random.choice(range(1, N + 1)) for i in range(self.seq_length)], dtype=torch.int64)
            result = torch.cat((torch.zeros(self.delay), seq[:self.seq_length - self.delay])).type(torch.int64)
            yield seq, result

DELAY = 4
DATASET_SIZE = 200000
ds = EchoDataset(delay=DELAY, size=DATASET_SIZE)

D = DELAY
def test_model(model, sequence_length=15):
    """
    This is the test function that runs 100 different strings through your model,
    and checks the error rate.
    """
    total = 0
    correct = 0
    for i in range(500):
        s = ''.join([random.choice(string.ascii_lowercase) for i in range(random.randint(15, 25))])
        result = model.test_run(s)
        for c1, c2 in zip(s[:-D], result[D:]):
            correct += int(c1 == c2)
            if int(c1 == c2) != 1:
                print(s[:-4], result[4:])
        total += len(s) - D

    return correct / total



'''
Author: Xiang Pan
Date: 2022-03-02 23:20:23
LastEditTime: 2022-03-03 22:22:57
LastEditors: Xiang Pan
Description: 
FilePath: /HW2/hw2_rnn.ipynb
@email: xiangpan@nyu.edu
'''
import torch
import wandb
class GRUMemory(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        #TODO: initialize your submodules
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(N+1,
                                self.hidden_size, 
                                num_layers=1,
                                batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, N+1)
        self.hidden = None

    def forward(self, x):
        # inputs: x - input tensor of shape (batch_size, seq_length, N+1)
        # returns:
        # logits (scores for softmax) of shape (batch size, seq_length, N + 1)
        # TODO implement forward pass
        # embedding = self.embedding(x)
        embedding = x.clone()
        # embedding[:, 4:, :] = x[ :, :-4, :]
        # embedding[:, 0:4, :] = 0
        gru_outs, _ = self.gru(embedding)
        logits = self.linear(gru_outs)
        return logits

    @torch.no_grad()
    def test_run(self, s):
        # This function accepts one string s containing lowercase characters a-z. 
        # You need to map those characters to one-hot encodings, 
        # then get the result from your network, and then convert the output 
        # back to a string of the same length, with 0 mapped to ' ', 
        # and 1-26 mapped to a-z.
        idx = s_to_idx(s)
        onehot = idx_to_onehot(torch.tensor(idx, dtype=torch.int64)).unsqueeze(0).cuda()
        logits = self.forward(onehot)
        pred = torch.argmax(logits, dim=2)
        pred = pred.squeeze(0)
        pred_str = ''.join([chr(c + ord('a') - 1) for c in pred])
        return pred_str

import time

start_time = time.time()


model = GRUMemory(hidden_size=128).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 30], gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()
# TODO
epoch = 0
test_acc = 0
while test_acc <= 0.99:
    ds = EchoDataset(delay=DELAY, size=DATASET_SIZE)
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=512)
    for x, y in train_dataloader:
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        model.train()
        optimizer.zero_grad()
        x = x.cuda()
        y = y.cuda()
        x_onehot = idx_to_onehot(x)
        y_onehot = idx_to_onehot(y)

        logits = model(x_onehot)
        logits = logits[:, 4:, :]
        y = y[:, 4:]

        loss = criterion(logits.reshape(-1, N + 1), y.reshape(-1))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        loss.backward()
        optimizer.step()

    test_acc = test_model(model)
    scheduler.step()
    print(f'epoch {epoch}: {test_acc}', f'loss: {loss.item()}')
    epoch += 1

end_time = time.time()
duration = end_time - start_time
accuracy = test_model(model)
assert duration < 600, 'execution took f{duration:.2f} seconds, which longer than 10 mins'
assert accuracy > 0.99, f'accuracy is too low, got {accuracy}, need 0.99'
print('tests passed')
