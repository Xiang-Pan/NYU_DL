'''
Author: Xiang Pan
Date: 2022-02-17 15:15:35
LastEditTime: 2022-02-17 15:42:32
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1/mlp_tester.py
@email: xiangpan@nyu.edu
'''
import torch
import torch.nn as nn


class TMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(2, 20)
        self.f = nn.ReLU()
        self.linear2 = nn.Linear(20, 7)
        self.g = nn.ReLU()

    def forward(self, x):
        self.z1 = self.linear1(x)
        self.z1.retain_grad()
        self.z2 = self.f(self.z1)
        self.z2.retain_grad()
        self.z3 = self.linear2(self.z2)
        self.z3.retain_grad()
        self.y_hat = self.g(self.z3)
        self.y_hat.retain_grad()
        return self.y_hat


import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP, mse_loss, bce_loss

net = MLP(
    linear_1_in_features=2,
    linear_1_out_features=20,
    f_function='relu',
    linear_2_in_features=20,
    linear_2_out_features=7,
    g_function='relu'
)
x = torch.randn(3, 2)
y = torch.randn(3, 7)

net.clear_grad_and_cache()
y_hat = net.forward(x)
J, dJdy_hat = mse_loss(y, y_hat)
net.backward(dJdy_hat)

model = TMLP()

model.linear1.weight.data = net.parameters['W1']
model.linear1.bias.data = net.parameters['b1']
model.linear2.weight.data = net.parameters['W2']
model.linear2.bias.data = net.parameters['b2']

y_hat_autograd = model(x)
J_autograd = F.mse_loss(y_hat_autograd, y)
print(J == J_autograd)

model.zero_grad()
J_autograd.backward()

# print(model.y_hat.grad)
print(model.z3.grad)
# print(net.grads['dJdz3'])




