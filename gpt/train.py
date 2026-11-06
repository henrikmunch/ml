import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(1337)

def header(title="", char="=", width=60):
    print(f' {title} '.center(width, char) if title else char * width)

# hyperparameters
block_size = 8  # aka maximal context length
batch_size = 4  # number independent sequences processed in parallel
n_embed = 32
max_iters = 3000
eval_iters = 200
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# read file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

header("Length of dataset in characters")
print(len(text))

# unique characters in the text
chars = sorted(set(text))
vocab_size = len(chars)

header("Characters")
print(''.join(chars))
header('Size of vocabulary')
print(vocab_size)

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

def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# self-attention

class Head(nn.Module):
    tril: torch.Tensor

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, C = x.shape  # (B, T, C)
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

# bigram

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits
        # for the next token from a lookup TabError
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both a (B,T) tensor of int
        tok_emb = self.token_embedding_table(idx)  # (B, T, C1 = n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C1)
        x = tok_emb + pos_emb
        logits = self.lm_head(x)  # (B, T, C2 = vocab_size)

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
            logits, _ = self(idx)
            # focus only on last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled text to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

header("Loss")
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
header("Generate from model:")
print(decode(m.generate(context, max_new_tokens=50)[0].tolist()))
