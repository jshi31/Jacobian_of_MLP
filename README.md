# Jacobian of MLP

This repository provide a fast Jacobian Matrix calculation of MLP in Pytorch.

## Introduction

Computing the Jacobian matrix of neural network is not supported by autograd framework, e.g. Tensorflow and Pytorch, because their autograd only support scalar output for neural network. So the major solution is loop over all the element of the output and concatenate each indivisual gradient. However, such method is of low efficiency, because there are usually batch dimension and channel dimension in the output tensor, e.g., shape (batch_size, dim_size), which is computatinally wasting to loop every element. 

This repository provide an explicit Jacobian Matrix calculation of MLP. Note that it is not computed by autograd, so the Jacobian matrix can still be used to contribute the loss and further optimize the network parameter using autograd.

## Usage

### Check its effect

```shell
python mlp.py
```

then you will observe the gradient calculate by autograd is the same as our result in Jacobian.

### General useage

```python
import torch
from mlp import MLP
mlp = MLP(n_inputs=3, n_outputs=2, n_layers=4, n_units=20)
# create input
x = torch.rand(2, 3) # batch_size 2, n_inputs 3
# Usage 1: compute the output
y = mlp(x) #  y: (batch_size 2, n_outputs 2)
# Usage 2: compute the jacobian matrix dy/dx.
J = mlp.jacobian(x)  # (batch_size 2, n_outputs 2, n_inputs)
# Note: Usage1 and Usage2 are independent.

# Good features: Use J as a part in network training.
# For example: add panelty to J: Det(J^TJ) and train.
loss = (J.permute(0, 2, 1)@J).det().mean()
loss.backward()
```



