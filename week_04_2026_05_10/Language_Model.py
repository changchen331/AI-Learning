from pathlib import Path

LENGTH = 30

DATA_PATH = Path("./data")


def load_data(datapath=DATA_PATH):
    texts = []
    for filepath in datapath.glob("*.txt"):
        with open(filepath, encoding="utf-8") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(texts):
    chars = sorted(set(texts))
    char2id = dict(zip(chars, range(len(chars))))
    id2char = dict(zip(range(len(chars)), chars))
    return char2id, id2char


if __name__ == '__main__':
    text = load_data()
    Char2Id, Id2Char = build_vocab(text)
    print(f"Loaded {len(text)} characters")
    print(f"Vocabulary size: {len(Char2Id)}")
    print(text)
    print(Char2Id)
    print(Id2Char)
