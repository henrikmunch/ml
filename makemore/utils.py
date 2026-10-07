import random
import torch

g = torch.Generator().manual_seed(2147483647)

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))

char_to_i = {char:i+1 for i,char in enumerate(chars)} # i+1 because i=0 reserved for '.'
char_to_i['.'] = 0
i_to_char = {i:char for char,i in char_to_i.items()}

block_size = 3 # context length: how many characters do we take to predict the next?
vocab_size = len(i_to_char) # number of characters 

n_embed = 10 # dimensionality of the character embedding vectors
n_hidden = 100 # number of neurons in the hidden layer of the MLP

max_steps = 20_000
batch_size = 32

def build_dataset(words):
    X, Y = [], []

    for w in words:

        context = [0] * block_size # initialize with all '...'
        for char in w + '.':
            ix = char_to_i[char]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    # An element of X contains three (indices for) characters, like [e, m, m]
    X = torch.tensor(X)
    # An element of Y contains the (index for) the next char, like [a] (as in emma)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

# Train (80%) split, dev/valuation split (10%), test split (10%)
Xtr, Ytr   = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte   = build_dataset(words[n2:])