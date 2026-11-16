import torch

def attention_step(decoder_state, encoder_states):
    alignment_score = decoder_state @ encoder_states.transpose(1, 2)
    weights = torch.softmax(alignment_score, dim=-1)
    context = weights @ encoder_states
    return context, weights


encoder_states = torch.randn(32, 12, 128)
decoder_state = torch.randn(32, 1, 128)
context, weights = attention_step(decoder_state, encoder_states)
print(context.shape, weights.shape)
print(weights[0, 0].sum())  # Verify the 12 attention weights sum to 1.0!