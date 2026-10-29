import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# read file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f'Length of dataset in characters: {len(text)}')

# unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f'Characters: {''.join(chars)}')
print(f'Size of vocabulary: {vocab_size}')

# create a mapping from chars to ints
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]  # encoder: string -> list of ints
decode = lambda li: ''.join([itos[i] for i in li])  # decoder: list<int> -> str

# encode dataset and store it as a torc.tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split data into train and val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# aka maximal context length
block_size = 8

block_size_bool = False
if block_size_bool:
    x = train_data[:block_size]
    y = train_data[1:block_size + 1]
    for t in range(block_size):
        context = x[:t + 1]
        target = y[t]
        print(f'when input is {context} the target is {target}')

# number independent sequences processed in parallel
batch_size = 4


def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# there are batch_size * block_size = 32 examples in a single batch
xb, yb = get_batch('train')
print('intputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

# bigram


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits
        # for the next token from a lookup TabError
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both a (B,T) tensor of int
        # logits has dimension (B=batch,T=time,C=channel=vocab_size)
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # to follow pytorch conventions
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled text to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
            return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
idx_test = m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)
print(logits.shape)
print(loss)
print(decode(idx_test[0].tolist()))





