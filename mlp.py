import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
        """ The MLP must have the first and last layers as FC.
        :param n_inputs: input dim
        :param n_outputs: output dim
        :param n_layers: layer num = n_layers + 2
        :param n_units: the dimension of hidden layers
        :param nonlinear: nonlinear function
        """
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_units = n_units
        self.nonlinear = nonlinear
        self.inv_nonlinear = self.get_inv_nonliner()
        # create layers
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))
        self.layers = layers

    def get_inv_nonliner(self):
        """
        This will return the inverse of the nonlinear function, which is with input as the activation rather than z
        Currently only support sigmoid and tanh.
        """
        if self.nonlinear == nn.Tanh:
            inv = lambda x: 1 - x * x
        elif self.nonlinear == nn.Sigmoid:
            inv = lambda x: x * (1 - x)
        else:
            assert False, '{} inverse function is not emplemented'.format(self.nonlinear)
        return inv

    def forward(self, x):
        for layer_i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def jacobian(self, x):
        """
        :param x: (bs, n_inputs)
        :return: J (bs, n_outputs, n_inputs)
        """
        bs = x.shape[0]
        # 1. forward pass and get all inverse activation
        inv_activations = []
        # first do forward
        for layer_i, layer in enumerate(self.layers):
            x = layer(x)
            if layer_i % 2 == 1: # is activation
                inv_activations.append(self.inv_nonlinear(x))
        # 2. compute error in DP fashion
        len_layers = len(self.layers)
        len_Deltas = (len_layers + 1) // 2
        for Delta_i in range(len_Deltas - 1, -1, -1):
            if Delta_i == len_Deltas - 1:  # if at the final layer, assign it as unit matrix
                Delta = torch.diag(torch.ones(self.n_outputs, device=x.device)).unsqueeze(0).\
                    expand(bs, self.n_outputs, self.n_outputs)
            else:
                layer_i = Delta_i * 2
                W = self.layers[layer_i + 2].weight  # current Delta use the previous W
                inv_activation_i = Delta_i
                inv_activation = inv_activations[inv_activation_i]
                Delta = Delta @ (W.unsqueeze(0) * inv_activation.unsqueeze(1))
        # 3. obtain solution with
        W = self.layers[0].weight
        J = Delta @ W.unsqueeze(0).expand(bs, self.n_units, self.n_inputs)
        return J


if __name__ == '__main__':
    x = torch.rand(2, 3, requires_grad=True)  # (bs, n_inputs)
    mlp = MLP(3, 2, n_layers=4, n_units=20)

    # show the gradient respectively
    y = mlp(x)  #(bs, n_outputs)
    y[0, 0].backward(retain_graph=True)
    print('grad', x.grad)
    x.grad = None

    y[0, 1].backward(retain_graph=True)
    print('grad', x.grad)
    x.grad = None

    y[1, 0].backward(retain_graph=True)
    print('grad', x.grad)
    x.grad = None

    y[1, 1].backward(retain_graph=True)
    print('grad', x.grad)
    x.grad = None


    J = mlp.jacobian(x)
    print('Jacobian', J)
