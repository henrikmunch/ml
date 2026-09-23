words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))

char_to_i = {char:i+1 for i,char in enumerate(chars)} # i+1 because i=0 reserved for '.'
char_to_i['.'] = 0
i_to_char = {i:char for char,i in char_to_i.items()}