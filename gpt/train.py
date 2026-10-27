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
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] # encoder: string --> list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: list<ints> --> string

