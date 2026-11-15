import torch

def cosine_similarity(u, v):
    dot = torch.sum(u * v, dim=-1)
    u_norm = torch.linalg.norm(u, dim=-1)
    v_norm = torch.linalg.norm(v, dim=-1)
    return dot / (u_norm * v_norm + 1e-8)

u = torch.tensor([3.0, 4.0])
v = torch.tensor([6.0, 8.0])
#print(cosine_similarity(u, v))


vocab = {
    "man":   torch.tensor([0.0, 0.0, 0.8]),
    "woman": torch.tensor([0.0, 1.0, 0.8]),
    "king":  torch.tensor([1.0, 0.0, 0.9]),
    "queen": torch.tensor([1.0, 1.0, 0.9]),
    "apple": torch.tensor([0.1, 0.1, 0.1])
}

target = vocab['king'] - vocab['man'] + vocab['woman']
for word, vec in vocab.items():
    sim = cosine_similarity(target, vec)
    if torch.isclose(sim, torch.tensor(1.0)):
        print(f'{word} and at {sim}')