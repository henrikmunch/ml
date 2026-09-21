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


# Sample names from the bigram 


P = N.float(); 
# Sum across rows to normalize the probability distribution 
# The division uses broadcasting: dim=(27,27) divided by dim=(27,1), with the '1' getting stretched to 27
P /= P.sum(dim=1, keepdim=True) 

g = torch.Generator().manual_seed(29833)
out = []
ix = 0
while True:
    ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
    out.append(i_to_char[ix])
    if ix == 0: # End-token (recall char_to_i['<.>'] = 0)
        break
print(''.join(out))