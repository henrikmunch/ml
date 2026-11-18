import itertools

# Byte-pair encoding

text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokens = text.encode("utf-8")  # raw bytes
tokens = list(map(int, tokens))  # convert to a list of integers in range 0..255 for convenience

vocab_size = 276  # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens)  # copy so we don't destroy the original list


def get_stats(ids):
    counts = {}
    for pair in itertools.pairwise(ids):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
    newids = []
    i = 0
    while i < len(ids):
        # if we are not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


merges = {}  # (int, int) -> int
for i in range(num_merges):
    stats = get_stats(ids)
    top_pair = max(stats, key=lambda p: stats[p])
    idx = 256 + i  # mint new token
    print(f"merging {top_pair} into a new token {idx}")
    ids = merge(ids, top_pair, idx)
    merges[top_pair] = idx


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]


def decode(ids):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


def encode(text):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


# GPT series
