class Sequential:
    def __init__(self, layers): # takes the dict of layer.
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers: # for each layer, we will pass the input x through it and it sends out the input to the next layer which is a activation function.
            x = layer(x)
            self.out = x
        return self.out
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]