import torch

class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)   # [W_aa | W_ax]
        self.h2o = torch.nn.Linear(hidden_size, output_size)  # [W_ya]

    def forward(self, x):
        B, T, C = x.shape
        a = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            x_t = x[:, t, :]
            ax_t = torch.cat([a , x_t], dim=1)
            a = torch.tanh(self.i2h(ax_t))
        return self.h2o(a)

x = torch.randn(32, 10, 64)
rnn = SimpleRNN(input_size=64, hidden_size=128, output_size=10)
logits = rnn(x)
print(logits.shape)      