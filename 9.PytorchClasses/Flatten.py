import torch
class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    def parameters(self):
        return []


#A bottleneck in how neural networks learn patterns. Going straight to (27, 80) from (27, 8, 10) squashes the entire context together all at once, 
# whereas going through intermediate steps like (27, 4, 20) creates a hierarchical tree structure (often called a WaveNet structure).
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n      #n is the number of consecutive elements you want to merge together
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)    # The new sequence length becomes T divided by n (T // n) and The new feature size becomes C multiplied by n (C * n).
        # Optimization: If the sequence length (T) is reduced down to exactly 1, squeeze out that dimension to convert the 3D tensor (B, 1, New_C) into a 2D tensor (B, New_C).
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self):
        return []