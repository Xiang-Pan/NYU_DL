'''
Author: Xiang Pan
Date: 2022-02-16 17:10:10
LastEditTime: 2022-02-17 16:13:10
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1/test7.py
@email: xiangpan@nyu.edu
'''
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP, mse_loss, bce_loss

net = MLP(
    linear_1_in_features=2,
    linear_1_out_features=20,
    f_function='sigmoid',
    linear_2_in_features=20,
    linear_2_out_features=5,
    g_function='relu'
)
x = torch.randn(10, 2)
y = (torch.randn(10, 5) < 0.5) * 1.0

net.clear_grad_and_cache()
y_hat = net.forward(x)
J, dJdy_hat = mse_loss(y, y_hat)
net.backward(dJdy_hat)

#------------------------------------------------
# compare the result with autograd
net_autograd = nn.Sequential(
    OrderedDict([
        ('linear1', nn.Linear(2, 20)),
        ('sigmoid1', nn.Sigmoid()),
        ('linear2', nn.Linear(20, 5)),
        ('relu2', nn.ReLU()),
    ])
)
net_autograd.linear1.weight.data = net.parameters['W1']
net_autograd.linear1.bias.data = net.parameters['b1']
net_autograd.linear2.weight.data = net.parameters['W2']
net_autograd.linear2.bias.data = net.parameters['b2']

y_hat_autograd = net_autograd(x)

J_autograd = torch.nn.MSELoss()(y_hat_autograd, y)

net_autograd.zero_grad()
J_autograd.backward()

print((net_autograd.linear1.weight.grad.data - net.grads['dJdW1']).norm() < 1e-3)
print((net_autograd.linear1.bias.grad.data - net.grads['dJdb1']).norm() < 1e-3)
print((net_autograd.linear2.weight.grad.data - net.grads['dJdW2']).norm() < 1e-3)
print((net_autograd.linear2.bias.grad.data - net.grads['dJdb2']).norm()< 1e-3)
#------------------------------------------------
