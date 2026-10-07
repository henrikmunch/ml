from utils import *
import torch
import torch.nn.functional as F

# Parameters 

# Normalizations come from gain / sqrt(fan_in)
C  = torch.randn((vocab_size, n_embed), generator=g) 
W1 = torch.randn((n_embed * block_size, n_hidden), generator=g) * (5/3) / (n_embed * block_size)**0.5
b1 = torch.randn(n_hidden, generator=g) * (5/3) / n_hidden**0.5
W2 = torch.randn((n_hidden, vocab_size), generator=g) * (5/3) / n_hidden**0.5
b2 = torch.rand(vocab_size, generator=g) * 0 

# Batch normalization gain and bias
bn_gain = torch.ones((1, n_hidden))
bn_bias = torch.zeros((1, n_hidden))
bn_mean_running = torch.zeros((1, n_hidden))
bn_std_running = torch.ones((1, n_hidden))

parameters = [C, W1, b1, W2, b2, bn_gain, bn_bias]
for p in parameters:
    p.requires_grad = True

# Explanation of emb = C[X]

    # Let C = torch.randn((27, 2))
    # C is the embedding matrix, sending each of the vocab_size=27 chars to a (say) n_embed=2 dimensional embedding vector
    #
    # X: 32 examples
    # │
    # ├── [0, 0, 5]
    # ├── [0, 5, 13]
    # ├── [5, 13, 13]
    # │
    # └── ...
    # 
    # ↓ Embedding
    # 
    # emb: 32 examples
    # │
    # ├── [[0.2, -1.1], [0.2, -1.1], [0.4, 0.8]]
    # ├── [[0.2, -1.1], [0.4, 0.8], [-0.9, 0.4]]
    # ├── [[0.4, 0.8], [-0.9, 0.4], [-0.9, 0.4]]
    # │
    # └── ...

# Build a NN that takes the X's and predicts the Y's

loss_i = []
for i in range(max_steps):

    # Mini batch: list of batch_size many random indices between [0, num of rows in X (aka num of examples)]
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # ----------------------
    # Forward pass
    # ----------------------
    emb = C[Xb] # grabbing the randomly chosen rows of X
    # The .view re-interprets (32, 3, 2) as (32, 6)
    # The '-1' in .view instructs torch to infer the '32' from the shape of W
    emb_cat = emb.view(emb.shape[0], -1)

    # Linear layer
    h_pre_act = emb_cat @ W1 + b1 # hidden layer pre-activation. can actually remove b1 due to batch norm

    # Batch norm layer
    bn_mean_i = h_pre_act.mean(0, keepdim=True)
    bn_std_i = h_pre_act.std(0, keepdim=True)
    h_pre_act = bn_gain * (h_pre_act - bn_mean_i) / bn_std_i + bn_bias # batch normalization

    with torch.no_grad():
        bn_mean_running = 0.999 * bn_mean_running + 0.001 * bn_mean_i
        bn_std_running = 0.999 * bn_std_running + 0.001 * bn_std_i

    # Non-linearity
    h = torch.tanh(h_pre_act) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb)

    # ----------------------
    # Backward pass
    # ----------------------
    for p in parameters:
        p.grad = None
    loss.backward()

    # ----------------------
    # Update parameters
    # ----------------------
    lr = 0.1 if i < 10_000 else 0.01 
    for p in parameters:
        p.data += -lr * p.grad

    # Track stats
    if i % 1000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    loss_i.append(loss.log10().item())

# Sample from the model

sample_bool = False
if sample_bool == True:

    g = torch.Generator().manual_seed(2147483647 + 10)
    
    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])] # dimensions = (N_examples=1, block_size, n_embed)
            h = torch.tanh(emb.view(1,-1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0: # if end character .
                break
            
        print(''.join(i_to_char[i] for i in out))

# Print loss for training, validation or test sets

@torch.no_grad # disables gradient tracking
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x]
    emb_cat = emb.view(emb.shape[0], -1)
    h_pre_act = emb_cat @ W1 + b1 
    h_pre_act = bn_gain * (h_pre_act - bn_mean_running) / bn_std_running + bn_bias 
    h = torch.tanh(h_pre_act)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')
