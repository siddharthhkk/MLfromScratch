import torch
# linear layer -> y = W * A.T + b
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):  # when using batch normalization we don't need bias so thats why option to false it!
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5  # in the example above fan_in is block_size * embedding_size and fan_out is hidden_size
        if bias == True: self.bias = torch.zeros(fan_out)
        else: self.bias = None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None: self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])