import torch

class Sigmoid:
    """
    Sigmoid activation function
    """
    # @staticmethod
    def forward(self, x):
        """
        Args:
            x: the input tensor

        Return:
            y: the output tensor
        """
        return 1 / (1 + torch.exp(-x))
    
    def backward(self, dJdy):
        """
        Args:
            dJdy: the gradient of the loss with respect to the output of this layer

        Return:
            dJdx: the gradient of the loss with respect to the input of this layer
        """
        y = self.forward(self.x)
        return dJdy.mul(y.mul(1 - y))

class ReLU:
    """
    ReLU activation function
    """
    # @staticmethod
    def forward(self, x):
        """
        Args:
            x: the input tensor

        Return:
            y: the output tensor
        """
        x[x<0] = 0
        return x
    
    def backward(self, dJdy):
        """
        Args:
            dJdy: the gradient of the loss with respect to the output of this layer

        Return:
            dJdx: the gradient of the loss with respect to the input of this layer
        """
        y = self.forward(self.x)
        return dJdy.mul(y > 0)

class Identity:
    """
    Identity activation function
    """
    def forward(self, x):
        """
        Args:
            x: the input tensor

        Return:
            y: the output tensor
        """
        return x
    
    def backward(self, dJdy):
        """
        Args:
            dJdy: the gradient of the loss with respect to the output of this layer

        Return:
            dJdx: the gradient of the loss with respect to the input of this layer
        """
        return dJdy

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = self.get_activation_function(f_function)
        self.g_function = self.get_activation_function(g_function)

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )

        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def get_activation_function(self, function_name):
        if function_name.lower() == "relu":
            return ReLU()
        elif function_name.lower() == "sigmoid":
            return Sigmoid()
        elif function_name.lower() == "identity":
            return Identity()
        else:
            raise ValueError("Unknown activation function: {}! Please use relu|sigmoid|identity".format(function_name))

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        # batch first input
        # batch_size = x.shape[0]
        self.cache['x'] = x
        z1 = x @ self.parameters['W1'].T + self.parameters['b1']
        self.cache['z1'] = z1

        z2 = self.f_function.forward(z1)
        self.cache['z2'] = z2

        z3 = z2 @ self.parameters['W2'].T + self.parameters['b2']
        self.cache['z3'] = z3

        y_hat = self.g_function.forward(z3)
        self.cache['y_hat'] = y_hat

        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        # calculate gradients
        batch_size = dJdy_hat.shape[0]
        
        dJdy_hat = dJdy_hat / batch_size / self.parameters['W2'].shape[0] 
        # print("dJdy_hat:\n ", dJdy_hat)

        self.f_function.x = self.cache['z1']
        
        self.g_function.x = self.cache['z3']
        
        dJdz3 = self.g_function.backward(dJdy_hat) # (b, k)

        dJdW2 = dJdz3.T @ self.cache['z2'] # (b, k)^T @ (b, L1) = (k, L1)
        dJdb2 = dJdz3.sum(axis=0)          # (b, L1) -> (L1)

        dJdz2 = dJdz3 @ self.parameters['W2']   # (b, k) @ (k, L1) = (b, L1)
        dJdz1 = self.f_function.backward(dJdz2) # (b, L1) * f'

        # z1 = x @ W1.T + b1
        dJdW1 =  dJdz1.T @ self.cache['x'] # (b, L1)^T @ (b, n) = (L1, n)
        dJdb1 = dJdz1.sum(axis=0)          # (b, L1) -> (L1)

        self.grads['dJdW1'] += dJdW1
        self.grads['dJdb1'] += dJdb1 
        self.grads['dJdW2'] += dJdW2 
        self.grads['dJdb2'] += dJdb2 
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    loss = torch.mean(torch.pow((y_hat - y), 2))
    dJdy_hat = 2 * (y_hat - y) 
    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    y_hat = y_hat.double()
    y = y.double()
    y_hat = torch.clamp(y_hat, 1e-12, 1 - 1e-12)
    loss = -torch.mean(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
    dJdy_hat = (y_hat - y) / (y_hat * (1 - y_hat))

    return loss.float(), dJdy_hat.float()
