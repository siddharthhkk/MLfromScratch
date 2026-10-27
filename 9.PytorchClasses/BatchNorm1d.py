import torch
class BatchNorm1d:
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # learnable params
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        #running buffers
        self.mean_running = torch.zeros(dim)
        self.var_running = torch.ones(dim)

    def __call__(self, x):
        if self.training is True:  # if training is True then we will calculate the mean and std of the current batch and update the running mean and std
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True) 
        else: #if not training we want to use the running mean and std for inference
            xmean = self.mean_running
            xvar = self.var_running
        xhat = (x - xmean) / (xvar + self.eps) ** 0.5
        self.out = self.gamma * xhat  + self.beta
        if self.training: 
            with torch.no_grad():
                self.mean_running = (1 - self.momentum) * self.mean_running  + self.momentum * xmean
                self.var_running = (1 - self.momentum) * self.var_running + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

# it turns ouut that batch normalization is not very effective for small batch sizes and it is also not very effective for recurrent neural networks. 
# so in prod people use layer normalization instead of batch normalization. Layer normalization normalizes the activations across the features instead of across the batch.
# for it we just need to change the mean and var from mean(0) and std(0) to mean(1) and var(-1, keepdim=True) so it always grabs the last feature dimension even on 3D Transformer tensors
# this is the exact difference between normalizing down the batch column vs. normalizing across each individual example's feature row. And because LayerNorm computes 
# the mean and variance across each single example's features on the fly, it has zero dependency on other examples in the batch—which means you delete self.training, 
# mean_running, and var_running completely.


# also group normalization is also used in prod which normalizes the activations across the groups of features instead of across the batch or across the features.
# still batch normalization is used in prod for convolutional neural networks and fully connected networks.
# and after all this data normalization before training is always important regardless of the normalization technique used in the model.