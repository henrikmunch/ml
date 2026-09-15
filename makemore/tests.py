import torch

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))

char_to_i = {char:i+1 for i,char in enumerate(chars)} # i+1 because i=0 reserved for '.'
char_to_i['<.>'] = 0
i_to_char = {i:char for char,i in char_to_i.items()}

# Bigram dimension from 26 alphabetical letters 1 special start/end char '.'
dim_b = 26 + 1
N = torch.zeros((dim_b, dim_b), dtype=torch.int32)

for w in words:
    chs = ['<.>'] + list(w) + ['<.>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = char_to_i[ch1]
        ix2 = char_to_i[ch2]
        N[ix1, ix2] += 1

p = N[0].float()
p = p / p.sum()
print(p)
g = torch.Generator().manual_seed(29834)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print(ix)
print(i_to_char[ix])