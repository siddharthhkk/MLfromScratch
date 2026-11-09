import torch

def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + alpha, min=0.0)
    return loss.mean()

def gram_matrix(features):
    C, H, W = features.shape
    X = features.view(C, H*W)
    return X @ X.T

A = torch.randn(32, 128)
P = torch.randn(32, 128)
N = torch.randn(32, 128)
print(triplet_loss(A, P, N))
feat = torch.randn(64, 28, 28)
print(gram_matrix(feat).shape)