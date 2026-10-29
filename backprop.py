import torch
import torch.nn.functional as F
words = ["grok", "hello", "world", "python", "machine", "learning", "neural", "network", "backpropagation", "gradient", "descent", "optimization", 
"algorithm", "data", "science", "artificial", "intelligence", "deep", "reinforcement", "supervised", "unsupervised", "semi-supervised", "transfer", 
"representation", "embedding", "regularization", "dropout", "batch", "normalization", "activation", "function", "loss", "cross", "entropy", "mean", 
"squared", "error", "stochastic", "gradient", "descent", "momentum", "adam", "rmsprop", "learning", "rate", "scheduler", "early", "stopping", "overfitting", 
"underfitting", "bias", "variance", "tradeoff", "hyperparameter", "tuning", "grid", "search", "random", "search", "bayesian", "optimization", "model",
"selection", "evaluation", "metrics", "accuracy", "precision", "recall", "f1", "score", "roc", "auc", "confusion", "matrix", "cross-validation", "k-fold", 
"leave-one-out", "stratified", "sampling", "imbalanced", "dataset", "synthetic", "augmentation", "feature", "engineering", "dimensionality", "reduction", 
"pca", "lda", "tsne"]

char = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i,s in enumerate(char)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

embedding_size = 10
hidden_size = 200
block_size = 4
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y
import random 
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xv, Yv = build_dataset(words[n1:n2])
Xt, Yt = build_dataset(words[n2:])

C = torch.randn((27, embedding_size))
W1 = torch.randn((block_size * embedding_size, hidden_size)) * (5 / 3) / ((block_size * embedding_size) ** 0.5) # Kaiming initialization
b1 = torch.randn(hidden_size) * 0.01
W2 = torch.randn((hidden_size, len(stoi))) * 0.01
b2 = torch.randn(len(stoi)) * 0.01

gamma = torch.ones((1, hidden_size)) * 0.1 + 1.0 #bngain
beta = torch.zeros((1, hidden_size)) #bnbias

parameters = [C, W1, b1, W2, b2, gamma, beta]
for p in parameters:
    p.requires_grad = True


# to compare manual gradient to pytorch loss.grad!
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')



batch_size = 32
ix = torch.randint(0, Xtr.shape[0], (batch_size,))
Xb, Yb = Xtr[ix], Ytr[ix]

# forward pass (broken into more variables as we will come backward!)
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
# linear layer 1
hprebn = embcat @ W1 + b1
#batchnorm layer
bnmeani = (1 / batch_size) * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = (1/(batch_size-1)) * bndiff2.sum(0, keepdim=True)   # in paper its just 1 / m(or if doing with batchs btch_size) but we take 1 / (m - 1) as in paper during inference bessel's corr.
bnvar_inv = (bnvar + 1e-5) ** 0.5
bnraw = bndiff * bnvar_inv
hpreact = gamma * bnraw + beta
# non - linearity
h = torch.tanh(hpreact)

# linear layer 2
logits = h @ W2 + b2
#cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes 
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum**-1    # if using (1.0 / counts_sum) then can't get backprop to be bit exact
probs = counts * counts_sum_inv # basically softmax
logprobs = probs.log()
loss = -logprobs[range(batch_size), Yb].mean()

#backward pass
for p in parameters:
    p.grad = None
for t in [logprobs, probs, counts, counts_sum_inv, counts_sum, norm_logits, logit_maxes, logits, h, hpreact, bnraw, bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hprebn, embcat, emb]:
    t.retain_grad()
loss.backward()

#1 backprop through the whole forward pass manually
dlogprobs = torch.zeros_like(logprobs)   # doing dloss/ dlogprobs
# basically we have to set the gradient of the correct class to -1/batch_size and all other classes to 0 as dloss/dlogprobs = -1/batch_size * 1 for the correct class and 0 for all other classes
dlogprobs[range(batch_size), Yb] = -1/batch_size
cmp('dlogprobs', dlogprobs, logprobs)
dprobs = (1 / probs) * dlogprobs
cmp('dprobs', dprobs, probs)

dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True) 
cmp('dcounts_sum_inv', dcounts_sum_inv, counts_sum_inv)
dcounts_sum = -(counts_sum ** -2) * dcounts_sum_inv
cmp('dcounts_sum', dcounts_sum, counts_sum)
dcounts = counts_sum_inv * dprobs   
dcounts += torch.ones_like(counts) * dcounts_sum # as counts is used twice so we have to add the gradients from both the paths one from dcounts_sum_inv and one from dprobs.
cmp('dcounts' , dcounts, counts)
dnorm_logits = norm_logits.exp() * dcounts
cmp('dnorm_logits', dnorm_logits, norm_logits)
dlogit_maxes = -1 * dnorm_logits.sum(1, keepdim=True) 
cmp('dlogit_maxes', dlogit_maxes, logit_maxes)
dlogits = 1 * dnorm_logits   # as logits is used twice so we have to add the gradients from both the paths one from dnorm_logits and one from dlogit_maxes.
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes 
# we are using one_hot encoding to get the gradient of the max value in logits as dlogit_maxes is the gradient of the max value in logits and we have to add it to the gradient of logits.
# (so one hot will make all the max places 1 and all other places 0 and then we multiply it with dlogit_maxes to get the gradient of logits)
cmp('dlogits', dlogits, logits)

dh = dlogits @ W2.T
cmp('dh', dh, h)
dW2 = h.T @ dlogits
cmp('dW1', dW2, W2)
db2 = dlogits.sum(0)
cmp('db2', db2, b2)
dhpreact = (1 - (torch.tanh(hpreact)**2)) * dh
cmp('dhpreact', dhpreact, hpreact)
dbnraw = gamma * dhpreact
cmp('dbnraw', dbnraw, bnraw)
dgamma = (bnraw * dhpreact).sum(0, keepdim=True)
cmp('dgamma', dgamma, gamma)
dbeta = 1 * dhpreact.sum(0, keepdim=True)
cmp('dbeta', dbeta, beta)
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
cmp('dbnvar_inv', dbnvar_inv, bnvar_inv)
dbnvar = 0.5 * (bnvar + 1e-5) ** -0.5 * dbnvar_inv
cmp('dbnvar', dbnvar, bnvar)

dbndiff = bnvar_inv * dbnraw
dbndiff2 = (1 / (batch_size - 1)) * torch.ones_like(bndiff2) * dbnvar
cmp('dbndiff2', dbndiff2, bndiff2)
dbndiff += 2 * (bndiff) * dbndiff2
cmp('dbndiff', dbndiff, bndiff)

dhprebn = 1 * dbndiff
dbnmeani = -1 * dbndiff.sum(0, keepdim=True)
cmp('dbnmeani', dbnmeani, bnmeani)
dhprebn += (1 / batch_size) * torch.ones_like(hprebn) * dbnmeani
cmp('dhprebn', dhprebn, hprebn)
dembcat = dhprebn @ W1.T
cmp('dembcat', dembcat, embcat)
dW1 = embcat.T @ dhprebn
cmp('dW1', dW1, W1)
db1 = 1 * dhprebn.sum(0)
cmp('db1', db1, b1)
demb = dembcat.view(emb.shape)
cmp('demb', demb, emb)
dC = torch.zeros_like(C)
for i in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[i, j]
        dC[ix] += demb[i, j] 
cmp('dC', dC, C)

#2 backprop through cross_entropy but all in one go

# forward pass before:
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # subtract max for numerical stability
# counts = norm_logits.exp()
# counts_sum = counts.sum(1, keepdims=True)
# counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
# probs = counts * counts_sum_inv
# logprobs = probs.log()
# loss = -logprobs[range(n), Yb].mean()

# now:
loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), 'diff:', (loss_fast - loss).item())

#backward pass
d2logits = F.softmax(logits, 1)
d2logits[range(batch_size), Yb] -= 1
d2logits /= batch_size
cmp('d2logits', d2logits, logits)
#dlogits are basically the gradient which pull ups the correct class and push down the incorrect classes. 
# The gradient is scaled by 1/batch_size to account for the averaging in the loss function. and its sum is 0 as push and pull cancels out.
d2logits.sum()


#3 backprop through batchnorm but all in one go
# forward pass

# before:
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = gamma * bnraw + beta

# now:
hpreact_fast = gamma * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + beta
print('max diff:', (hpreact_fast - hpreact).abs().max())
n = batch_size

# backward pass

# before we had:
# dbnraw = bngain * dhpreact
# dbndiff = bnvar_inv * dbnraw
# dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
# dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
# dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
# dbndiff += (2*bndiff) * dbndiff2
# dhprebn = dbndiff.clone()
# dbnmeani = (-dbndiff).sum(0)
# dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)

#now:
dhprebn = gamma*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))

cmp('hprebn', dhprebn, hprebn) # I can only get approximate to be true, my maxdiff is 9e-10


#4  putting it all together
n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((len(stoi), n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1
# Layer 2
W2 = torch.randn((n_hidden, len(stoi)),          generator=g) * 0.1
b2 = torch.randn(len(stoi),                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

max_steps = 200000
batch_size = 32
n = batch_size
lossi = []

with torch.no_grad():
  for i in range(max_steps):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

    # forward pass
    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    # Linear layer
    hprebn = embcat @ W1 + b1 # hidden layer pre-activation
    # BatchNorm layer
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    # Non-linearity
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
      p.grad = None
    #loss.backward()

    # manual backprop! #swole_doge_meme
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n
    # 2nd layer backprop
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    # tanh
    dhpreact = (1.0 - h**2) * dh
    # batchnorm backprop
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
    # 1st layer
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    # embedding
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
      for j in range(Xb.shape[1]):
        ix = Xb[k,j]
        dC[ix] += demb[k,j]
    grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p, grad in zip(parameters, grads):
      #p.data += -lr * p.grad     # old way (using PyTorch grad from .backward())
      p.data += -lr * grad

    # track stats
    if i % 10000 == 0: # print every once in a while
      print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
  #   if i >= 100:
  #     break



  # measure the mean/std over the entire training set
with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    bnmean = hpreact.mean(0, keepdim=True)
    bnvar = hpreact.var(0, keepdim=True, unbiased=True)

# evaluate train and val loss
@torch.no_grad()
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xv, Yv),
    'test': (Xt, Yt),
  }[split]
  emb = C[x]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1 + b1
  hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())
split_loss('train')
split_loss('val')


# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
      # forward pass:
      emb = C[torch.tensor([context])]
      embcat = emb.view(emb.shape[0], -1)
      hpreact = embcat @ W1 + b1
      hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
      h = torch.tanh(hpreact)
      logits = h @ W2 + b2
      # Sample
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    print(''.join(itos[i] for i in out))