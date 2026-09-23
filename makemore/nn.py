import torch
import torch.nn.functional as F
from utils import words, char_to_i, i_to_char

# Create bigram training set (x,y)
# Given first character, x, of the bigram and want to predict second character, y
xs, ys =  [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = char_to_i[ch1]
        ix2 = char_to_i[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# 27 = 26 alphabetical letters + 1 special character '.'
xenc = F.one_hot(xs, num_classes=27).float()
yenc = F.one_hot(ys, num_classes=27).float()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True) # Weights


# Train the NN (which just has one layer W) with gradient descent 


for i in range(20):
    # Forward pass
    logits = xenc @ W # Log-counts
    counts = logits.exp() # Softmax. Equivalient to N in bigram.py
    probs = counts / counts.sum(dim=1, keepdim=True) # Probability of next character
    reg = (W**2).mean() # Regularization, which keeps W entries close to 0 
    loss = - probs[torch.arange(num), ys].log().mean() + reg

    # Backward pass
    W.grad = None # Reset gradients to zero (otherwise they accumulate on each pass)
    loss.backward()

    # Update parameters (the prefactor is the learning rate)
    W.data += -50 * W.grad

    # print(f'{i}: {loss}')


# Sample from the NN
 
     
for i in range(5):
    out = [] 
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp() 
        p = counts / counts.sum(dim=1, keepdim=True) 

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(i_to_char[ix])
        if ix == 0: # End-token (recall char_to_i['.'] = 0)
            break
    print(''.join(out))